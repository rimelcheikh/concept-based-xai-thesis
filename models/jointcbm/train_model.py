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
from models.jointcbm.utils import make_tfds, set_seeds, write_setting, numpy_from_dl

from torchvision import transforms
from PIL import Image

AUTOTUNE = tf.data.AUTOTUNE


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("Could not set memory growth:", e)

PER_BACKBONE = {
    "mobilenetv2": ("sgd", 1e-3),
    "resnet50": ("adam", 2e-4),
    "efficientnetb0": ("adam", 1e-4),
    "inceptionv3": ("adam", 1e-4),
}

# -------------------------
# Preprocess helper (FIX)
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


def to_255_float(x: np.ndarray) -> np.ndarray:
    """Ensure float32 and pixel scale 0..255 before tf.keras.applications preprocess_input."""
    x = x.astype("float32")
    if x.size > 0 and np.nanmax(x) <= 1.5:
        x = x * 255.0
    return x


# -------------------------
# PIL-augment helpers
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
    ce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=label_smoothing,
                                                 reduction=tf.keras.losses.Reduction.NONE)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_oh = tf.one_hot(y_true, depth=K, dtype=tf.float32)   # smoothing handled inside CE
        per_sample = ce(y_oh, y_pred)                          # [B]
        w = tf.gather(cw, y_true)                              # [B]
        return tf.reduce_mean(per_sample * w)

    return loss_fn


# -------------------------
# Backbone builder
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
# Model + training
# -------------------------
def build_joint_cbm_model(M, K, backbone, img_size, weights, train_backbone):
    """
    Joint CBM:
      image -> sigmoid concepts -> class softmax
    """
    input_shape = (img_size, img_size, 3)
    inp = tf.keras.Input(shape=input_shape, name="image")

    base = build_backbone(backbone, input_shape, weights=weights, train_backbone=train_backbone)

    # IMPORTANT FIX: do NOT hard-wire training=train_backbone (BatchNorm correctness)
    x = base(inp)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    concepts = tf.keras.layers.Dense(M, activation="sigmoid", name="concepts")(x)
    class_out = tf.keras.layers.Dense(K, activation="softmax", name="class")(concepts)

    return tf.keras.Model(inputs=inp, outputs=[concepts, class_out])


def train_one_run_cbm(
    X_train, y_train, a_train,
    X_val, y_val, a_val,
    X_test, y_test, a_test,
    M, K,
    backbone, img_size,
    batch_size, n_epochs,
    optimizer_name, lr,
    dataset_name,
    train_backbone,
    backbone_weights,
    coeff_concept=1.0,
    coeff_class=1.0,
    save_dir=None,
    seed=42,
):
    preprocess_fn = get_backbone_preprocess(backbone)

    train_ds = make_tfds(X_train, y_train, a_train, batch_size,
                         shuffle=True, cache=False, seed=seed, reshuffle_each_iteration=False)
    val_ds   = make_tfds(X_val, y_val, a_val, batch_size,
                         shuffle=False, cache=False, seed=seed, reshuffle_each_iteration=False)
    test_ds  = make_tfds(X_test, y_test, a_test, batch_size,
                         shuffle=False, cache=False, seed=seed, reshuffle_each_iteration=False)

    # preprocess per-batch inside tf.data (same ops as to_255_float + preprocess_input),
    # then pack targets (ab, yb)
    def _map_preprocess_and_pack(xb, yb, ab):
        xb = tf.cast(xb, tf.float32)
        xb_max = tf.reduce_max(xb)
        xb = tf.cond(xb_max <= 1.5, lambda: xb * 255.0, lambda: xb)
        xb = preprocess_fn(xb)
        return xb, (ab, yb)

    train_ds = train_ds.map(_map_preprocess_and_pack, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds   = val_ds.map(_map_preprocess_and_pack, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds  = test_ds.map(_map_preprocess_and_pack, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    model = build_joint_cbm_model(M, K, backbone, img_size, backbone_weights, train_backbone)
    opt = create_optimizer(optimizer_name, lr)

    # dataset-specific concept supervision
    if dataset_name in ["aPY","CUB"]:
        concept_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    else:
        concept_loss = tf.keras.losses.MeanSquaredError()

    class_weights = compute_class_weights(y_train, K)
    cls_loss = make_weighted_label_smoothed_ce(K, class_weights, label_smoothing=0.05)

    model.compile(
        optimizer=opt,
        loss={
            "concepts": concept_loss,
            "class": cls_loss,
        },
        loss_weights={"concepts": coeff_concept, "class": coeff_class},
        metrics={
            "class": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),]
        },
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
    val_bal_cb = FullValBalancedAcc(val_ds=val_ds,
                                    K=K,
                                    get_probs_fn=lambda model, batch: (
                                        batch[1][1].numpy(),  # y_true = yb
                                        model(batch[0], training=False)[1].numpy(),  # class_probs
                                    ))

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=n_epochs,
        callbacks=[val_bal_cb, ckpt_best, csv_logger, es],
        verbose=2,
    )

    # IMPORTANT: force-load best checkpoint (same protocol as CNN)
    if os.path.exists(best_path):
        model.load_weights(best_path)
    else:
        print("WARNING: best checkpoint not found at", best_path)

    # test eval: class balanced accuracy
    y_preds, y_true = [], []
    for xb, (ab, yb) in test_ds:
        _, class_probs = model(xb, training=False)
        y_preds.append(tf.argmax(class_probs, axis=1).numpy())
        y_true.append(yb.numpy())

    y_pred = np.concatenate(y_preds, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    acc = float((y_pred == y_true).mean())
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=K, dtype=tf.float32).numpy()
    tp = np.diag(cm)
    row_sum = cm.sum(axis=1)
    recall_c = np.divide(tp, row_sum, out=np.zeros_like(tp, dtype=float), where=row_sum > 0)
    mask = row_sum > 0
    bal_acc = float(recall_c[mask].mean() if mask.any() else 0.0)

    model.save_weights(os.path.join(save_dir, "final.weights.h5"))
    return bal_acc, {"accuracy": acc, "balanced_accuracy": bal_acc, "confusion_matrix": cm}


def run_all(args):
    seed = 42
    set_seeds(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = args.dataset
    backbone = args.model

    backbone_weights = None if args.weights == "None" else "imagenet"

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
            args.batch_size, args.data_dir
        )

        if annotations is None:
            raise ValueError(f"{dataset}: annotations is None but Joint-CBM requires concept supervision.")

        if images.max() <= 1.01:
            base_images = (images * 255.0).astype("uint8")
        else:
            base_images = images.astype("uint8")

        labels = np.asarray(labels).reshape(-1).astype(int)
        annotations = np.asarray(annotations).astype("float32")

        M = int(annotations.shape[1])
        K = int(np.max(labels)) + 1

        train_pil_aug = extract_pil_augs(train_tf)
        test_pil_aug = extract_pil_augs(test_tf)

        kf = StratifiedKFold(n_splits=args.runs, shuffle=True, random_state=seed)

    elif dataset == "CUB":
        data_name = "CUB"
        result = get_dataset_loaders(
            data_name, data_config, seed, args.save_dir,
            args.batch_size, args.data_dir
        )
        train_dl, val_dl, test_dl, _, concepts, classes, concept_map, label_to_idx_awa = result

        X_train_full, y_train_full, a_train_full = numpy_from_dl(train_dl)
        X_val_full, y_val_full, a_val_full = numpy_from_dl(val_dl)
        X_test_full, y_test_full, a_test_full = numpy_from_dl(test_dl)

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

        a_train_full = np.asarray(a_train_full).astype("float32")
        a_val_full = np.asarray(a_val_full).astype("float32")
        a_test_full = np.asarray(a_test_full).astype("float32")

        M = int(a_train_full.shape[1])
        K = len(classes)

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
    print(f"JointCBM | dataset={dataset} | backbone={backbone} | opt={optimizer_name} | lr={lr} | img_size={img_size}")
    print("==============================\n")

    if dataset in ["AwA2", "aPY"]:
        fold_id = 0
        for train_idx, test_idx in kf.split(base_images, labels):
            fold_id += 1

            if args.only_fold is not None and fold_id != args.only_fold:
                continue

            run_dir = os.path.join(
                args.save_dir,
                f"{dataset}_{backbone}/{args.epochs}_{args.batch_size}_{optimizer_name}_{lr}/"
                f"{args.coeff_concept}_{args.coeff_class}/fold{fold_id}"
            )
            os.makedirs(run_dir, exist_ok=True)

            done_flag = os.path.join(run_dir, "final.weights.h5")
            if os.path.exists(done_flag):
                print(f"[SKIP] JointCBM | {dataset} | {backbone} | fold {fold_id} already done")
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

            a_train = annotations[tr_idx]
            a_val   = annotations[val_idx]
            a_test  = annotations[test_idx]

            X_train_aug = apply_pil_augs_np(X_train_raw, train_pil_aug, img_size)
            X_val_aug   = apply_pil_augs_np(X_val_raw,   test_pil_aug,  img_size)
            X_test_aug  = apply_pil_augs_np(X_test_raw,  test_pil_aug,  img_size)

            X_train = X_train_aug.astype(np.uint8)
            X_val   = X_val_aug.astype(np.uint8)
            X_test  = X_test_aug.astype(np.uint8)

            write_setting(
                run_dir, backbone, dataset, dataset,
                args.coeff_concept, args.coeff_class, None,
                fold_id, args.runs, optimizer_name, lr, args.batch_size,
                None, None,
                args.val_split, args.epochs, len(y_train), len(y_val), len(y_test), M, K
            )

            test_bal_acc, _ = train_one_run_cbm(
                X_train, y_train, a_train,
                X_val, y_val, a_val,
                X_test, y_test, a_test,
                M, K,
                backbone, img_size,
                args.batch_size, args.epochs,
                optimizer_name, lr,
                dataset,
                args.train_backbone,
                backbone_weights,
                coeff_concept=args.coeff_concept,
                coeff_class=args.coeff_class,
                save_dir=run_dir,
                seed=seed,
            )

            print(f"[DONE] JointCBM | {dataset} | {backbone} | fold {fold_id} | test bal acc: {test_bal_acc:.4f}")

            tf.keras.backend.clear_session()
            gc.collect()

    else:
        for fold_id in range(1, args.runs + 1):
            if args.only_fold is not None and fold_id != args.only_fold:
                continue
            run_dir = os.path.join(
                args.save_dir,
                f"{dataset}_{backbone}/{args.epochs}_{args.batch_size}_{optimizer_name}_{lr}/"
                f"{args.coeff_concept}_{args.coeff_class}/fold{fold_id}"
            )
            os.makedirs(run_dir, exist_ok=True)

            done_flag = os.path.join(run_dir, "final.weights.h5")
            if os.path.exists(done_flag):
                print(f"[SKIP] JointCBM | {dataset} | {backbone} | fold {fold_id} already done")
                continue

            run_seed = seed + fold_id - 1
            set_seeds(run_seed)
            import torch
            torch.manual_seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)

            X_train = apply_pil_augs_np(X_train_full, train_pil_aug, img_size).astype(np.uint8)
            X_val   = apply_pil_augs_np(X_val_full,   test_pil_aug,  img_size).astype(np.uint8)
            X_test  = apply_pil_augs_np(X_test_full,  test_pil_aug,  img_size).astype(np.uint8)

            y_train = y_train_full
            y_val   = y_val_full
            y_test  = y_test_full

            a_train = a_train_full
            a_val   = a_val_full
            a_test  = a_test_full

            write_setting(
                run_dir, backbone, dataset, dataset,
                args.coeff_concept, args.coeff_class, None,
                fold_id, args.runs, optimizer_name, lr, args.batch_size,
                None, None,
                None, args.epochs, len(y_train), len(y_val), len(y_test), M, K
            )

            test_bal_acc, _ = train_one_run_cbm(
                X_train, y_train, a_train,
                X_val, y_val, a_val,
                X_test, y_test, a_test,
                M, K,
                backbone, img_size,
                args.batch_size, args.epochs,
                optimizer_name, lr,
                dataset,
                args.train_backbone,
                backbone_weights,
                coeff_concept=args.coeff_concept,
                coeff_class=args.coeff_class,
                save_dir=run_dir,
                seed=run_seed,
            )

            print(f"[DONE] JointCBM | {dataset} | {backbone} | fold {fold_id} | test bal acc: {test_bal_acc:.4f}")

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
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val_split", type=float, default=0.2)

    parser.add_argument("--override_hparams", action="store_true", default=False)

    parser.add_argument("--coeff_concept", type=float, default=5.0)
    parser.add_argument("--coeff_class", type=float, default=5.0)

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
