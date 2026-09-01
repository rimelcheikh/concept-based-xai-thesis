import os
import gc

# -------------------------
# Determinism OFF (MUST be before TF import)
# -------------------------
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "0"
os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, Callback

from data_loaders.data_loader import get_dataset_loaders
from data_loaders.configs import dataset_config
from models.cnn.utils import make_tfds, set_seeds, write_setting, numpy_from_dl

from torchvision import transforms
from PIL import Image

AUTOTUNE = tf.data.AUTOTUNE


# -------------------------
# PACBM hyperparam selection
# -------------------------
PER_BACKBONE = {
    "mobilenetv2": ("sgd", 1e-3),
    "resnet50": ("adam", 2e-4),
    "efficientnetb0": ("adam", 1e-4),
    "inceptionv3": ("adam", 1e-4),
}


# -------------------------
# GPU memory growth
# -------------------------
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("Could not set memory growth:", e)


# -------------------------
# Preprocess helper
# -------------------------
def get_backbone_preprocess(backbone_name: str):
    name = backbone_name.lower()
    if name == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input
    if name == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input
    if name == "inceptionv3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        return preprocess_input
    if name == "efficientnetb0":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        return preprocess_input
    return lambda x: x


# -------------------------
# PIL augment helpers
# -------------------------
def extract_pil_augs(torch_transform):
    if isinstance(torch_transform, transforms.Compose):
        pil_transforms = []
        for t in torch_transform.transforms:
            if isinstance(t, transforms.ToTensor):
                break
            if isinstance(t, transforms.Normalize):
                continue
            pil_transforms.append(t)
        return transforms.Compose(pil_transforms) if len(pil_transforms) else None
    return torch_transform


def apply_pil_augs_np(X, pil_transform, image_size):
    X_out = []
    for x in X:
        img = Image.fromarray(x.astype(np.uint8))
        if pil_transform is not None:
            img = pil_transform(img)
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.BILINEAR)
        x_aug = np.array(img, dtype=np.uint8)
        if x_aug.ndim == 2:
            x_aug = np.stack([x_aug] * 3, axis=-1)
        X_out.append(x_aug)
    return np.stack(X_out, axis=0)


# -------------------------
# Metrics
# -------------------------
def _balanced_accuracy_from_cm(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(np.float64)
    row_sum = cm.sum(axis=1).astype(np.float64)
    recall_c = np.divide(tp, row_sum, out=np.zeros_like(tp), where=row_sum > 0)
    mask = row_sum > 0
    return float(recall_c[mask].mean() if mask.any() else 0.0)


class FullValBalancedAcc(Callback):
    """
    Compute epoch-level balanced accuracy on the FULL validation set using a confusion matrix.
    Writes logs["val_bal_acc"] so ModelCheckpoint/EarlyStopping can monitor it.
    """

    def __init__(self, val_ds, K: int, get_probs_fn, name="val_bal_acc"):
        super().__init__()
        self.val_ds = val_ds
        self.K = int(K)
        self.get_probs_fn = get_probs_fn
        self.name = name

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        cm = np.zeros((self.K, self.K), dtype=np.int64)

        for batch in self.val_ds:
            y_true, probs = self.get_probs_fn(self.model, batch)
            y_true = np.asarray(y_true).reshape(-1).astype(np.int32)
            y_pred = np.argmax(probs, axis=1).astype(np.int32)

            cm_batch = tf.math.confusion_matrix(
                y_true, y_pred, num_classes=self.K, dtype=tf.int32
            ).numpy()
            cm += cm_batch

        bal = _balanced_accuracy_from_cm(cm)
        logs[self.name] = bal
        print(f" — {self.name}: {bal:.5f}")


def compute_class_weights(y: np.ndarray, K: int) -> np.ndarray:
    """
    PACBM-style class weights:
      cc[k] = count of class k
      w[k] = sum(cc) / (K * cc[k])
      then normalize so mean(w)=1
    """
    y = np.asarray(y).reshape(-1).astype(np.int32)
    cc = np.bincount(y, minlength=K).astype(np.float32)
    cc = np.maximum(cc, 1.0)
    w = cc.sum() / (float(K) * cc)
    w = w / np.mean(w)
    return w.astype(np.float32)


def make_weighted_label_smoothed_ce(K: int, class_weights: np.ndarray, label_smoothing: float = 0.05):
    """
    Returns a loss(y_true_int, y_pred_prob) that:
      - one-hots y_true
      - applies label smoothing
      - applies per-sample weights based on class_weights[y_true]
      - returns mean weighted CE
    """
    cw = tf.constant(class_weights, dtype=tf.float32)
    ce = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=label_smoothing,
        reduction=tf.keras.losses.Reduction.NONE,
    )

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_oh = tf.one_hot(y_true, depth=K, dtype=tf.float32)
        per_sample = ce(y_oh, y_pred)
        w = tf.gather(cw, y_true)
        return tf.reduce_mean(per_sample * w)

    return loss_fn


# -------------------------
# Backbone + optimizer
# -------------------------
def build_backbone(backbone_name, input_shape, weights="imagenet", train_backbone=True):
    name = backbone_name.lower()
    if name == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(include_top=False, weights=weights, input_shape=input_shape)
    elif name == "resnet50":
        base = tf.keras.applications.ResNet50(include_top=False, weights=weights, input_shape=input_shape)
    elif name == "inceptionv3":
        base = tf.keras.applications.InceptionV3(include_top=False, weights=weights, input_shape=input_shape)
    elif name == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights=weights, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    base.trainable = bool(train_backbone)
    return base


def create_optimizer(optimizer_name, lr):
    name = optimizer_name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    opt = tf.keras.optimizers.get(optimizer_name)
    if hasattr(opt, "learning_rate"):
        opt.learning_rate = lr
    return opt


# -------------------------
# CNN model + training
# -------------------------
def build_cnn_model(K, backbone, img_size, weights, train_backbone):
    input_shape = (img_size, img_size, 3)
    inp = tf.keras.Input(shape=input_shape, name="image")
    base = build_backbone(backbone, input_shape, weights=weights, train_backbone=train_backbone)

    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(K, activation="softmax", name="class")(x)
    return tf.keras.Model(inputs=inp, outputs=out)


def train_one_run_cnn(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    K,
    backbone, img_size,
    batch_size, n_epochs,
    optimizer_name, lr,
    train_backbone,
    backbone_weights,
    save_dir,
    seed=42,
):
    dummy_a_train = np.zeros((len(y_train), 1), dtype=np.float32)
    dummy_a_val   = np.zeros((len(y_val), 1), dtype=np.float32)
    dummy_a_test  = np.zeros((len(y_test), 1), dtype=np.float32)

    train_ds = make_tfds(
        X_train, y_train, dummy_a_train, batch_size,
        shuffle=True, cache=False, seed=seed, reshuffle_each_iteration=False
    )
    val_ds = make_tfds(
        X_val, y_val, dummy_a_val, batch_size,
        shuffle=False, cache=False, seed=seed, reshuffle_each_iteration=False
    )
    test_ds = make_tfds(
        X_test, y_test, dummy_a_test, batch_size,
        shuffle=False, cache=False, seed=seed, reshuffle_each_iteration=False
    )

    preprocess_fn = get_backbone_preprocess(backbone)

    def _map_preprocess(xb, yb, ab):
        xb = tf.cast(xb, tf.float32)
        xb_max = tf.reduce_max(xb)
        xb = tf.cond(xb_max <= 1.5, lambda: xb * 255.0, lambda: xb)
        xb = preprocess_fn(xb)
        return xb, yb

    train_ds = train_ds.map(_map_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds   = val_ds.map(_map_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds  = test_ds.map(_map_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    debug_dir = os.path.join(save_dir, "debug_preprocessing")
    os.makedirs(debug_dir, exist_ok=True)
    np.save(os.path.join(debug_dir, "X_train_raw_first8.npy"), np.asarray(X_train[:8]))
    np.save(os.path.join(debug_dir, "y_train_first8.npy"), np.asarray(y_train).reshape(-1)[:8].astype(np.int32))
    for xb_dbg, yb_dbg in train_ds.take(1):
        np.save(os.path.join(debug_dir, "train_batch_post_preprocess.npy"), xb_dbg.numpy().astype(np.float32))
        np.save(os.path.join(debug_dir, "train_batch_y.npy"), yb_dbg.numpy().astype(np.int32))

    model = build_cnn_model(K, backbone, img_size, backbone_weights, train_backbone)
    opt = create_optimizer(optimizer_name, lr)

    class_weights = compute_class_weights(y_train, K)
    cls_loss = make_weighted_label_smoothed_ce(K, class_weights, label_smoothing=0.05)

    model.compile(
        optimizer=opt,
        loss=cls_loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    weights_dir = os.path.join(save_dir, "models")
    os.makedirs(weights_dir, exist_ok=True)

    best_path = os.path.join(weights_dir, "best_val_bal_acc.weights.h5")

    ckpt_best = ModelCheckpoint(
        filepath=best_path,
        monitor="val_bal_acc",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    csv_logger = CSVLogger(os.path.join(save_dir, "history.csv"), append=False)

    es = EarlyStopping(
        monitor="val_bal_acc",
        mode="max",
        patience=200,
        restore_best_weights=True,
    )

    val_bal_cb = FullValBalancedAcc(
        val_ds=val_ds,
        K=K,
        get_probs_fn=lambda model, batch: (
            batch[1].numpy(),
            model(batch[0], training=False).numpy()
        ),
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=n_epochs,
        callbacks=[val_bal_cb, ckpt_best, csv_logger, es],
        verbose=2,
    )

    if os.path.exists(best_path):
        model.load_weights(best_path)
    else:
        print("WARNING: best checkpoint not found at", best_path)

    y_preds, y_true = [], []
    for xb, yb in test_ds:
        probs = model(xb, training=False)
        y_preds.append(tf.argmax(probs, axis=1).numpy())
        y_true.append(yb.numpy())
    y_pred = np.concatenate(y_preds, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=K, dtype=tf.float32).numpy()
    tp = np.diag(cm)
    row_sum = cm.sum(axis=1)
    recall_c = np.divide(tp, row_sum, out=np.zeros_like(tp, dtype=float), where=row_sum > 0)
    mask = row_sum > 0
    bal_acc = float(recall_c[mask].mean() if mask.any() else 0.0)

    model.save_weights(os.path.join(save_dir, "final.weights.h5"))
    return bal_acc


# -------------------------
# Run-all loop
# -------------------------
def run_all(args):
    seed = 42
    set_seeds(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = args.dataset
    backbone = args.model

    backbone_weights = None if args.weights == "None" else "imagenet"


    if dataset == "aPY":
        batch_size = 64
    else:
        batch_size = args.batch_size

    data_config = dataset_config.get(dataset)

    if dataset in ["AwA2", "aPY"]:
        data_name = f"{dataset}_cv"

        (images, _,
            labels, _,
            annotations, _,
            mat_pd, _,
            concepts,
            label_to_idx_awa,
            mat_GT,
            train_tf,
            test_tf) = get_dataset_loaders(
            data_name, data_config, seed, args.save_dir,
            batch_size, args.data_dir
        )

        assert images.ndim == 4 and images.shape[-1] == 3, f"{dataset}: images expected [N,H,W,3], got {images.shape}"
        assert np.isfinite(images.astype(np.float32)).all(), f"{dataset}: images contain NaN/Inf"

        if images.max() <= 1.01:
            base_images = (images * 255.0).astype("uint8")
        else:
            base_images = images.astype("uint8")

        assert base_images.dtype == np.uint8
        assert 0 <= base_images.min() and base_images.max() <= 255

        labels = np.asarray(labels).reshape(-1).astype(int)
        K = int(np.max(labels)) + 1

        uniq = np.unique(labels)
        assert len(uniq) == K, f"{dataset}: labels not contiguous? K={K} unique={len(uniq)}"
        assert uniq.min() == 0 and uniq.max() == K - 1, f"{dataset}: expected labels 0..K-1, got {uniq.min()}..{uniq.max()}"

        train_pil_aug = extract_pil_augs(train_tf)
        test_pil_aug = extract_pil_augs(test_tf)

    elif dataset == "CUB":
        data_name = "CUB"
        result = get_dataset_loaders(data_name, data_config, seed, args.save_dir, batch_size, args.data_dir)
        train_dl, val_dl, test_dl, _, concepts, classes, concept_map, label_to_idx_awa = result

        X_train_full, y_train_full, _ = numpy_from_dl(train_dl)
        X_val_full, y_val_full, _ = numpy_from_dl(val_dl)
        X_test_full, y_test_full, _ = numpy_from_dl(test_dl)

        for nm, arr in [("X_train_full", X_train_full), ("X_val_full", X_val_full), ("X_test_full", X_test_full)]:
            arr = np.asarray(arr)
            assert arr.ndim == 4 and arr.shape[-1] in (1, 3), f"{dataset}: {nm} shape {arr.shape} unexpected"
            assert np.isfinite(arr.astype(np.float32)).all(), f"{dataset}: {nm} contains NaN/Inf"

        def _to_uint8(arr):
            arr = np.asarray(arr)
            if float(np.max(arr)) <= 1.01:
                return (arr * 255.0).astype("uint8")
            return np.clip(arr, 0, 255).astype("uint8")

        X_train_full = _to_uint8(X_train_full)
        X_val_full = _to_uint8(X_val_full)
        X_test_full = _to_uint8(X_test_full)

        y_train_full = np.asarray(y_train_full).reshape(-1).astype(int)
        y_val_full = np.asarray(y_val_full).reshape(-1).astype(int)
        y_test_full = np.asarray(y_test_full).reshape(-1).astype(int)

        K = len(classes)
        uniq = np.unique(np.concatenate([y_train_full, y_val_full, y_test_full]))
        assert uniq.min() == 0 and uniq.max() == K - 1, f"{dataset}: expected labels 0..K-1, got {uniq.min()}..{uniq.max()}"

        train_pil_aug = None
        test_pil_aug = None

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if backbone not in PER_BACKBONE:
        raise ValueError(f"Backbone {backbone} not in PER_BACKBONE mapping.")

    if args.override_hparams:
        optimizer_name = args.optimizer
        lr = args.lr
    else:
        optimizer_name, lr = PER_BACKBONE[backbone]

    img_size = args.img_size
    if backbone == "inceptionv3" and args.inception_img_size is not None:
        img_size = args.inception_img_size

    print("\n==============================")
    print(f"CNN | dataset={dataset} | backbone={backbone} | opt={optimizer_name} | lr={lr} | img_size={img_size}")
    print("==============================\n")

    if dataset in ["AwA2", "aPY"]:
        kf = StratifiedKFold(n_splits=args.runs, shuffle=True, random_state=seed)

        fold_id = 0
        for train_idx, test_idx in kf.split(base_images, labels):
            fold_id += 1

            if args.only_fold is not None and fold_id != args.only_fold:
                continue

            run_dir = os.path.join(
                args.save_dir,
                f"{dataset}_{backbone}/{args.epochs}_{batch_size}_{optimizer_name}_{lr}/fold{fold_id}"
            )
            os.makedirs(run_dir, exist_ok=True)

            done_flag = os.path.join(run_dir, "final.weights.h5")
            if os.path.exists(done_flag):
                print(f"[SKIP] {dataset} | {backbone} | fold {fold_id} already done")
                continue

            tr_idx, val_idx = train_test_split(
                train_idx,
                test_size=args.val_split,
                random_state=seed,
                stratify=labels[train_idx],
            )

            X_train_raw = base_images[tr_idx]
            X_val_raw   = base_images[val_idx]
            X_test_raw  = base_images[test_idx]

            y_train = labels[tr_idx]
            y_val   = labels[val_idx]
            y_test  = labels[test_idx]

            X_train_aug = apply_pil_augs_np(X_train_raw, train_pil_aug, img_size)
            X_val_aug   = apply_pil_augs_np(X_val_raw,   test_pil_aug,  img_size)
            X_test_aug  = apply_pil_augs_np(X_test_raw,  test_pil_aug,  img_size)

            X_train = X_train_aug.astype(np.uint8)
            X_val   = X_val_aug.astype(np.uint8)
            X_test  = X_test_aug.astype(np.uint8)

            write_setting(
                run_dir, backbone, dataset, dataset,
                None, None, None,
                fold_id, args.runs, optimizer_name, lr, batch_size,
                None, None,
                args.val_split, args.epochs, len(y_train), len(y_val), len(y_test), None, K
            )

            bal = train_one_run_cnn(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                K,
                backbone, img_size,
                batch_size, args.epochs,
                optimizer_name, lr,
                args.train_backbone,
                backbone_weights,
                run_dir,
                seed=seed,
            )

            print(f"[DONE] {dataset} | {backbone} | fold {fold_id} | test bal acc: {bal:.4f}")

            tf.keras.backend.clear_session()
            gc.collect()

    else:  # CUB
        for fold_id in range(1, args.runs + 1):
            if args.only_fold is not None and fold_id != args.only_fold:
                continue
            
            run_dir = os.path.join(
                args.save_dir,
                f"{dataset}_{backbone}/{args.epochs}_{batch_size}_{optimizer_name}_{lr}/fold{fold_id}"
            )
            os.makedirs(run_dir, exist_ok=True)

            done_flag = os.path.join(run_dir, "final.weights.h5")
            if os.path.exists(done_flag):
                print(f"[SKIP] {dataset} | {backbone} | fold {fold_id} already done")
                continue

            run_seed = seed + fold_id - 1
            set_seeds(run_seed)
            torch.manual_seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)

            X_train = apply_pil_augs_np(X_train_full, train_pil_aug, img_size)
            X_val   = apply_pil_augs_np(X_val_full,   test_pil_aug,  img_size)
            X_test  = apply_pil_augs_np(X_test_full,  test_pil_aug,  img_size)

            y_train = y_train_full
            y_val   = y_val_full
            y_test  = y_test_full

            write_setting(
                run_dir, backbone, dataset, dataset,
                None, None, None,
                fold_id, args.runs, optimizer_name, lr, batch_size,
                None, None,
                None, args.epochs, len(y_train), len(y_val), len(y_test), None, K
            )

            bal = train_one_run_cnn(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                K,
                backbone, img_size,
                batch_size, args.epochs,
                optimizer_name, lr,
                args.train_backbone,
                backbone_weights,
                run_dir,
                seed=run_seed,
            )

            print(f"[DONE] {dataset} | {backbone} | fold {fold_id} | test bal acc: {bal:.4f}")

            tf.keras.backend.clear_session()
            gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["AwA2", "aPY", "CUB"], default="aPY")
    parser.add_argument(
        "--model",
        type=str,
        choices=["mobilenetv2", "efficientnetb0", "inceptionv3"],
        default="mobilenetv2",
    )

    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.2)

    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--override_hparams", action="store_true", default=False)

    parser.add_argument("--save_dir", type=str, default="./trained_models/paper/baselines/")
    parser.add_argument("--data_dir", type=str, default="../data/")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--inception_img_size", type=int, default=None)

    parser.add_argument("--train_backbone", action="store_true", default=True)
    parser.add_argument("--no_train_backbone", dest="train_backbone", action="store_false")

    parser.add_argument("--weights", type=str, default="imagenet", choices=["imagenet", "None"])

    parser.add_argument("--only_fold", type=int, default=None, help="If set, run only this 1-based CV fold.")


    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    run_all(args)


if __name__ == "__main__":
    main()