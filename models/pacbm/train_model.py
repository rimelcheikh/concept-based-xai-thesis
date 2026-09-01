
import os
import gc


# DETERM FIX
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
import tensorflow as tf

from data_loaders.data_loader import get_dataset_loaders
from data_loaders.configs import dataset_config

from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, Callback

from models.pacbm.my_callbacks import ConfMatSaver, DynamicAnchorWarmup

from models.pacbm.model_pacbm import PACBModel

from models.pacbm.utils import make_tfds, get_backbone_preprocess, numpy_from_dl, set_seeds, write_setting

from torchvision import transforms
from PIL import Image

AUTOTUNE = tf.data.AUTOTUNE

#os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "2.0"


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # This will only print if something initialized the GPU earlier
        print("Could not set memory growth:", e)


# DETERM FIX
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism not fully enabled:", e)


# ---------------------------------------------------------
# Helpers to reuse PyTorch transforms
# ---------------------------------------------------------
def extract_pil_augs(torch_transform):
    """
    Take a torchvision.transforms.Compose that looks like:
      [ColorJitter, RandomResizedCrop, RandomHorizontalFlip,
       ToTensor, Normalize]
    and return a new Compose that keeps only the PIL-based augs:
      [ColorJitter, RandomResizedCrop, RandomHorizontalFlip]

    So Keras can handle normalization (preprocess_input) itself.
    """
    if isinstance(torch_transform, transforms.Compose):
        pil_transforms = []
        for t in torch_transform.transforms:
            # Stop at ToTensor: after that, pipeline expects tensors.
            if isinstance(t, transforms.ToTensor):
                break
            # Skip Normalize if it appears before ToTensor (just in case)
            if isinstance(t, transforms.Normalize):
                continue
            pil_transforms.append(t)

        if len(pil_transforms) == 0:
            return None
        return transforms.Compose(pil_transforms)

    # If it's not a Compose, assume it's already a PIL-level transform
    return torch_transform


def apply_pil_augs_np(X, pil_transform, image_size):
    """
    X: numpy array [N, H, W, 3] uint8
    pil_transform: PIL->PIL transform (ColorJitter, RandomResizedCrop, etc.)
    image_size: final target size

    Returns: numpy array [N, image_size, image_size, 3] uint8
    """
    X_out = []
    for x in X:
        img = Image.fromarray(x.astype(np.uint8))

        if pil_transform is not None:
            img = pil_transform(img)

        # Ensure final size in case transform pipeline isn't exact
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.BILINEAR)

        x_aug = np.array(img, dtype=np.uint8)
        if x_aug.ndim == 2:  # grayscale -> 3 channels
            x_aug = np.stack([x_aug] * 3, axis=-1)

        X_out.append(x_aug)

    return np.stack(X_out, axis=0)


# ---------------------------------------------------------
# Custom callbacks
# ---------------------------------------------------------
class EveryNEpochsCheckpoint(Callback):
    def __init__(self, save_dir, every_n=10, verbose=1):
        super().__init__()
        self.save_dir = save_dir
        self.every_n = every_n
        self.verbose = verbose
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        if epoch_num % self.every_n == 0:
            path = os.path.join(self.save_dir, f"epoch_{epoch_num:03d}.weights.h5")
            self.model.save_weights(path)
            if self.verbose:
                print(f"\n[Checkpoint] Saved weights to {path}")


class StopIfValBalAccTooLow(Callback):
    def __init__(self, threshold=0.1, epoch_limit=50, monitor="val_bal_acc", verbose=1):
        super().__init__()
        self.threshold = threshold
        self.epoch_limit = epoch_limit
        self.monitor = monitor
        self.verbose = verbose
        self.best_seen = float("-inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is not None:
            self.best_seen = max(self.best_seen, float(current))

        epoch_num = epoch + 1
        if epoch_num >= self.epoch_limit and self.best_seen <= self.threshold:
            if self.verbose:
                print(
                    f"\n[Early Stop] Best {self.monitor} after {epoch_num} epochs "
                    f"is {self.best_seen:.4f}, which did not exceed {self.threshold:.4f}. Stopping."
                )
            self.model.stop_training = True


# ---------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------
def create_optimizer(optimizer_name, lr):
    """
    Create a Keras optimizer from a string name and learning rate.

    Supports a few common optimizers explicitly, and falls back to
    tf.keras.optimizers.get for anything else.
    """
    if isinstance(optimizer_name, str):
        name = optimizer_name.lower()
    else:
        raise ValueError(f"Optimizer name must be a string, got {type(optimizer_name)}")

    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)

    if name == "adamw":
        if hasattr(tf.keras.optimizers, "AdamW"):
            return tf.keras.optimizers.AdamW(learning_rate=lr)
        raise ValueError("AdamW optimizer not available in this TensorFlow version.")

    try:
        opt = tf.keras.optimizers.get(optimizer_name)
        if hasattr(opt, "learning_rate"):
            opt.learning_rate = lr
        return opt
    except Exception as e:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'. Original error: {e}")


def compute_P_true_global(a_train, y_train, K, eps=1e-4):
    """
    a_train: [N, M] float in {0,1} or [0,1]
    y_train: [N] int labels in [0, K-1]
    returns: P_true_global [M, K]
    """
    a_train = np.asarray(a_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    N, M = a_train.shape

    one_hot = np.eye(K, dtype=np.float32)[y_train]  # [N, K]
    sum_attr = a_train.T @ one_hot  # (M, K)
    class_count = one_hot.sum(axis=0)  # (K,)
    class_count_safe = np.maximum(class_count, 1.0)

    P_true = sum_attr / class_count_safe[None, :]
    P_true = np.clip(P_true, eps, 1.0 - eps)
    return P_true, class_count


# ---------------------------------------------------------
# Model builder
# ---------------------------------------------------------
def build_and_compile(
    M,
    K,
    backbone,
    img_size,
    lr,
    dataset_name,
    train_backbone,
    coeff_l_a_CE,
    coeff_l_cls_CE,
    coeff_prior_anch,
    save_dir,
    class_count,
    P_true_global=None,
    backbone_weights="imagenet",
    use_ema_prior=False,
    ema_momentum=0.9,
    optimizer_name="adam",
):
    input_shape = (img_size, img_size, 3)
    model = PACBModel(
        input_size=input_shape,
        M=M,
        K=K,
        backbone_name=backbone,
        backbone_weights=backbone_weights,
        dataset_name=dataset_name,
        train_backbone=train_backbone,
        coeff_l_a_CE=coeff_l_a_CE,
        coeff_l_cls_CE=coeff_l_cls_CE,
        coeff_prior_anch=coeff_prior_anch,
        save_dir=save_dir,
        class_count=class_count,
        use_ema_prior=use_ema_prior,
        ema_momentum=ema_momentum,
    )

    dummy = (
        tf.zeros((1, img_size, img_size, 3), dtype=tf.float32),
        tf.zeros((1,), dtype=tf.int32),
        tf.zeros((1, M), dtype=tf.float32),
    )
    _ = model(dummy, training=False)

    if P_true_global is not None:

        def logit(p, eps=1e-4):
            p = np.clip(p, eps, 1.0 - eps)
            return np.log(p) - np.log(1.0 - p)

        gamma_init = logit(P_true_global).astype("float32")  # [M, K]
        model.class_layer.gamma.assign(gamma_init)

    optimizer = create_optimizer(optimizer_name, lr)
    model.compile(
        optimizer=optimizer,
        run_eagerly=False,
        weighted_metrics=[],
    )
    return model


# ---------------------------------------------------------
# Single run
# ---------------------------------------------------------
def train_one_run(
    X_train,
    y_train,
    a_train,
    X_val,
    y_val,
    a_val,
    X_test,
    y_test,
    a_test,
    M,
    K,
    backbone,
    img_size,
    lr,
    batch_size,
    n_epochs,
    coeff_l_a_CE,
    coeff_l_cls_CE,
    coeff_prior_anch,
    dataset_name,
    train_backbone,
    backbone_weights,
    save_dir,
    class_count,
    P_true_global=None,
    use_ema_prior=False,
    ema_momentum=0.9,
    start_mse=0.12,
    end_mse=0.06,
    optimizer_name="adam",
    seed=42,
):
    # --------- Pre-sanity (before preprocess_input) ----------
    def assert_image_block(name, x):
        x = np.asarray(x)
        assert x.ndim == 4, f"{name}: expected 4D [N,H,W,C], got shape {x.shape}"
        assert x.shape[-1] == 3, f"{name}: expected 3 channels last, got shape {x.shape}"
        assert x.dtype in (np.float32, np.float64, np.uint8, np.int32, np.int64), f"{name}: unexpected dtype {x.dtype}"
        assert np.isfinite(x.astype(np.float32)).all(), f"{name}: contains NaN/Inf"
        mx = float(np.max(x))
        mn = float(np.min(x))
        return mn, mx

    preprocess_fn = get_backbone_preprocess(backbone)

    # Assert raw ranges (should be 0..255 float/uint8 here)
    assert_image_block("X_train(pre)", X_train)
    assert_image_block("X_val(pre)", X_val)
    assert_image_block("X_test(pre)", X_test)

    # DO NOT preprocess full arrays (avoids huge float32 allocations / OOM).
    # We will preprocess per-batch inside tf.data using the exact same operations:
    #   cast->float32, optional *255 if <=1.5, then preprocess_input.
    def _map_preprocess(xb, yb, ab):
        xb = tf.cast(xb, tf.float32)
        xb_max = tf.reduce_max(xb)
        xb = tf.cond(xb_max <= 1.5, lambda: xb * 255.0, lambda: xb)
        xb = preprocess_fn(xb)
        return xb, yb, ab

    # Label/concept sanity
    y_train = np.asarray(y_train).reshape(-1).astype(np.int32)
    y_val = np.asarray(y_val).reshape(-1).astype(np.int32)
    y_test = np.asarray(y_test).reshape(-1).astype(np.int32)
    a_train = np.asarray(a_train).astype(np.float32)
    a_val = np.asarray(a_val).astype(np.float32)
    a_test = np.asarray(a_test).astype(np.float32)

    assert len(y_train) == X_train.shape[0] == a_train.shape[0], "Train sizes mismatch"
    assert len(y_val) == X_val.shape[0] == a_val.shape[0], "Val sizes mismatch"
    assert len(y_test) == X_test.shape[0] == a_test.shape[0], "Test sizes mismatch"
    assert a_train.ndim == 2 and a_train.shape[1] == M, f"a_train shape {a_train.shape}, expected [N,{M}]"
    assert np.all((y_train >= 0) & (y_train < K)), "y_train outside [0,K-1]"
    assert np.all((y_val >= 0) & (y_val < K)), "y_val outside [0,K-1]"
    assert np.all((y_test >= 0) & (y_test < K)), "y_test outside [0,K-1]"
    assert np.isfinite(a_train).all() and np.isfinite(a_val).all() and np.isfinite(a_test).all(), "annotations contain NaN/Inf"

    # --------- tf.data datasets ----------
    train_ds = make_tfds(
        X_train,
        y_train,
        a_train,
        batch_size,
        shuffle=True,
        cache=False,
        seed=seed,
        reshuffle_each_iteration=False,
    )
    val_ds = make_tfds(
        X_val,
        y_val,
        a_val,
        batch_size,
        shuffle=False,
        cache=False,
        seed=seed,
        reshuffle_each_iteration=False,
    )
    test_ds = make_tfds(
        X_test,
        y_test,
        a_test,
        batch_size,
        shuffle=False,
        cache=False,
        seed=seed,
        reshuffle_each_iteration=False,
    )

    # apply preprocessing per-batch (no big arrays created)
    train_ds = train_ds.map(_map_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(_map_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(_map_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    model = build_and_compile(
        M,
        K,
        backbone,
        img_size,
        lr,
        dataset_name,
        train_backbone,
        coeff_l_a_CE,
        coeff_l_cls_CE,
        coeff_prior_anch,
        save_dir,
        class_count,
        P_true_global=P_true_global,
        backbone_weights=backbone_weights,
        use_ema_prior=use_ema_prior,
        ema_momentum=ema_momentum,
        optimizer_name=optimizer_name,
    )

    cm_saver = ConfMatSaver(save_dir=save_dir, K=K, normalize_rows=True, save_every=1)

    weights_dir = os.path.join(save_dir, "models")
    os.makedirs(weights_dir, exist_ok=True)

    ckpt_best = ModelCheckpoint(
        filepath=os.path.join(weights_dir, "best_val_bal_acc.weights.h5"),
        monitor="val_bal_acc",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    ckpt_every_10 = EveryNEpochsCheckpoint(
        save_dir=weights_dir,
        every_n=300,
        verbose=1,
    )

    csv_logger = CSVLogger(os.path.join(save_dir, "history.csv"), append=False)

    dyn_anchor = DynamicAnchorWarmup(
        start_score=start_mse,
        end_score=end_mse,
        ema=0.9,
        verbose=1,
    )

    stop_if_low_val_bal_acc = StopIfValBalAccTooLow(
        threshold=0.1,
        epoch_limit=50,
        monitor="val_bal_acc",
        verbose=1,
    )

    es = EarlyStopping(
        monitor="val_bal_acc",
        mode="max",
        patience=200,
        restore_best_weights=True,
    )

    callbacks = [
        dyn_anchor,
        cm_saver,
        ckpt_best,
        ckpt_every_10,
        csv_logger,
        stop_if_low_val_bal_acc,
        es,
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=n_epochs,
        callbacks=callbacks,
        verbose=2,
    )

    all_preds, all_true = [], []
    for xb, yb, ab in test_ds:
        _, _, _, _, class_probs = model([xb, yb, ab], training=False)
        all_preds.append(tf.argmax(class_probs, axis=1).numpy())
        all_true.append(yb.numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    acc = (y_pred == y_true).mean()

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=K, dtype=tf.float32).numpy()
    tp = np.diag(cm)
    row_sum = cm.sum(axis=1)
    recall_c = np.divide(tp, row_sum, out=np.zeros_like(tp, dtype=float), where=row_sum > 0)
    mask = row_sum > 0
    bal_acc = recall_c[mask].mean() if mask.any() else 0.0

    test_acc = float(bal_acc)
    eval_res = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "confusion_matrix": cm,
    }

    model.save_weights(os.path.join(save_dir, "final.weights.h5"))
    return test_acc, eval_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c_annot", type=str, choices=["AwA2", "aPY", "CUB"], default="aPY")
    parser.add_argument("--dataset", type=str, choices=["CIFAR100", "AwA2", "aPY", "CUB"], default="aPY")
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet50", "mobilenetv2", "efficientnetb0", "inceptionv3"],
        default="mobilenetv2",
    )

    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=2e-05)
    parser.add_argument("--val_split", type=float, default=0.2)

    parser.add_argument("--coeff_l_a_CE", type=float, default=5.0)
    parser.add_argument("--coeff_l_cls_CE", type=float, default=5.0)
    parser.add_argument("--coeff_prior_anch", type=float, default=10.0)

    parser.add_argument("--start_mse", type=float, default=0.17)
    parser.add_argument("--end_mse", type=float, default=0.07)

    parser.add_argument("--use_ema_prior", action="store_true", default=True)
    parser.add_argument("--no_use_ema_prior", dest="use_ema_prior", action="store_false")
    parser.add_argument("--ema_momentum", type=float, default=0.9)

    parser.add_argument("--save_dir", type=str, default="./trained_models/TESTS/")
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--img_size", type=int, default=224)

    parser.add_argument("--train_backbone", action="store_true", default=True)
    parser.add_argument("--no_train_backbone", dest="train_backbone", action="store_false")

    parser.add_argument("--weights", type=str, default="imagenet", choices=["imagenet", "None"])

    parser.add_argument("--only_fold", type=int, default=None, help="If set, run only this 1-based CV fold.")

    args = parser.parse_args()

    seed = 42
    set_seeds(seed)

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = args.dataset
    backbone = args.model
    save_dir = args.save_dir 
    os.makedirs(save_dir, exist_ok=True)

    backbone_weights = None if args.weights == "None" else "imagenet"

    if dataset in ["AwA2", "aPY"]:
        data_name = f"{dataset}_cv"
        data_config = dataset_config.get(dataset)

        (
            images,
            _,
            labels,
            _,
            annotations,
            _,
            mat_pd,
            _,
            concepts,
            label_to_idx_awa,
            mat_GT,
            train_tf,
            test_tf,
        ) = get_dataset_loaders(data_name, data_config, seed, save_dir, args.batch_size, args.data_dir)

        labels = np.asarray(labels).reshape(-1).astype(int)
        annotations = np.asarray(annotations).astype(np.float32)

        if dataset.lower() == "apy":
            uniq = np.unique(annotations)
            assert np.all((annotations == 0.0) | (annotations == 1.0)), f"aPY annotations not binary, unique={uniq[:20]}"
        else:
            assert annotations.min() >= -1e-6 and annotations.max() <= 1.0 + 1e-6, "AwA2 annotations not in [0,1]"

        images = np.asarray(images)
        assert images.ndim == 4 and images.shape[-1] == 3, f"images shape {images.shape}, expected [N,H,W,3]"
        assert np.isfinite(images.astype(np.float32)).all(), "images contains NaN/Inf"

        if float(np.max(images)) <= 1.01:
            base_images = (images * 255.0).astype("uint8")
        else:
            base_images = images.astype("uint8")

        assert base_images.dtype == np.uint8
        assert base_images.ndim == 4 and base_images.shape[-1] == 3
        assert 0 <= base_images.min() and base_images.max() <= 255
        assert len(base_images) == len(labels) == len(annotations), "N mismatch between images/labels/annotations"

        M, K = mat_pd.shape[0], mat_pd.shape[1]

        train_pil_aug = extract_pil_augs(train_tf)
        test_pil_aug = extract_pil_augs(test_tf)

        kf = StratifiedKFold(n_splits=args.runs, shuffle=True, random_state=seed)

        fold_id = 0
        for train_idx, test_idx in kf.split(base_images, labels):
            fold_id += 1

            if args.only_fold is not None and fold_id != args.only_fold:
                continue

            tr_idx, val_idx = train_test_split(
                train_idx,
                test_size=args.val_split,
                random_state=seed,
                stratify=labels[train_idx],
            )

            X_train_raw = base_images[tr_idx]
            X_val_raw = base_images[val_idx]
            X_test_raw = base_images[test_idx]

            y_train, a_train = labels[tr_idx], annotations[tr_idx]
            y_val, a_val = labels[val_idx], annotations[val_idx]
            y_test, a_test = labels[test_idx], annotations[test_idx]

            img_size = args.img_size
            X_train_aug = apply_pil_augs_np(X_train_raw, train_pil_aug, img_size)
            X_val_aug = apply_pil_augs_np(X_val_raw, test_pil_aug, img_size)
            X_test_aug = apply_pil_augs_np(X_test_raw, test_pil_aug, img_size)

            assert X_train_aug.dtype == np.uint8 and X_train_aug.max() <= 255 and X_train_aug.min() >= 0
            assert X_val_aug.dtype == np.uint8 and X_val_aug.max() <= 255 and X_val_aug.min() >= 0
            assert X_test_aug.dtype == np.uint8 and X_test_aug.max() <= 255 and X_test_aug.min() >= 0

            # keep uint8 (do NOT convert to float32 arrays; avoids OOM)
            X_train = X_train_aug.astype(np.uint8)
            X_val = X_val_aug.astype(np.uint8)
            X_test = X_test_aug.astype(np.uint8)

            run_dir = os.path.join(
                save_dir,
                f"{dataset}_{backbone}/{args.epochs}_{args.batch_size}_{args.optimizer}_{args.lr}/"
                f"{args.coeff_l_a_CE}_{args.coeff_l_cls_CE}_{args.coeff_prior_anch}/"
                f"/{str(args.start_mse)}_{str(args.end_mse)}_{str(args.ema_momentum)}/"
                f"fold{fold_id}"
            )
            os.makedirs(run_dir, exist_ok=True)

            debug_dir = os.path.join(run_dir, "debug_preprocessing")
            os.makedirs(debug_dir, exist_ok=True)
            np.save(os.path.join(debug_dir, "X_train_uint8_first8.npy"), X_train_aug[:8])
            np.save(os.path.join(debug_dir, "X_val_uint8_first8.npy"), X_val_aug[:8])
            np.save(os.path.join(debug_dir, "X_test_uint8_first8.npy"), X_test_aug[:8])
            np.save(os.path.join(debug_dir, "y_train_first8.npy"), np.asarray(y_train).reshape(-1)[:8])
            np.save(os.path.join(debug_dir, "a_train_first8.npy"), np.asarray(a_train).astype(np.float32)[:8])

            write_setting(
                run_dir,
                backbone,
                dataset,
                dataset,
                args.coeff_l_a_CE,
                args.coeff_l_cls_CE,
                args.coeff_prior_anch,
                fold_id,
                args.runs,
                args.optimizer,
                args.lr,
                args.batch_size,
                args.start_mse,
                args.end_mse,
                args.val_split,
                args.epochs,
                len(y_train),
                len(y_val),
                len(y_test),
                M,
                K,
            )

            P_true_global, class_count_np = compute_P_true_global(a_train, y_train, K)
            class_count = tf.convert_to_tensor(class_count_np, dtype=tf.float32)

            final_weights = os.path.join(run_dir, "final.weights.h5")
            if os.path.exists(final_weights):
                print(f"[SKIP] Fold {fold_id} already completed")
                continue

            test_acc, _ = train_one_run(
                X_train,
                y_train,
                a_train,
                X_val,
                y_val,
                a_val,
                X_test,
                y_test,
                a_test,
                M,
                K,
                backbone,
                args.img_size,
                args.lr,
                args.batch_size,
                args.epochs,
                args.coeff_l_a_CE,
                args.coeff_l_cls_CE,
                args.coeff_prior_anch,
                args.dataset,
                args.train_backbone,
                backbone_weights,
                run_dir,
                class_count,
                P_true_global=P_true_global,
                use_ema_prior=args.use_ema_prior,
                ema_momentum=args.ema_momentum,
                start_mse=args.start_mse,
                end_mse=args.end_mse,
                optimizer_name=args.optimizer,
                seed=seed,
            )
            print(f"[Fold {fold_id}] Test balanced accuracy: {test_acc:.4f}")

            del X_train_raw, X_val_raw, X_test_raw
            del X_train_aug, X_val_aug, X_test_aug
            del X_train, X_val, X_test
            del y_train, y_val, y_test
            del a_train, a_val, a_test
            del P_true_global, class_count_np, class_count
            tf.keras.backend.clear_session()
            gc.collect()

    else:
        data_name = "CUB" if dataset == "CUB" else "cifar100"
        data_config = dataset_config[data_name]

        result = get_dataset_loaders(data_name, data_config, seed, save_dir, args.batch_size, args.data_dir)
        train_dl, val_dl, test_dl, _, concepts, classes, concept_map, label_to_idx_awa = result

        X_train, y_train, a_train = numpy_from_dl(train_dl)
        X_val, y_val, a_val = numpy_from_dl(val_dl)
        X_test, y_test, a_test = numpy_from_dl(test_dl)

        for nm, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
            arr = np.asarray(arr)
            assert arr.ndim == 4 and arr.shape[-1] in (1, 3), f"{nm} shape {arr.shape} unexpected"
            assert np.isfinite(arr.astype(np.float32)).all(), f"{nm} contains NaN/Inf"

        for name in ["X_train", "X_val", "X_test"]:
            arr = locals()[name]
            if float(np.max(arr)) <= 1.01:
                locals()[name] = (arr * 255.0).astype("float32")
            else:
                locals()[name] = arr.astype("float32")

        M = a_train.shape[1] if a_train is not None else len(concepts)
        K = len(classes)

        for fold_id in range(1, args.runs + 1):
            if args.only_fold is not None and fold_id != args.only_fold:
                continue

            run_seed = seed + fold_id - 1
            set_seeds(run_seed)
            import torch
            torch.manual_seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)

            run_dir = os.path.join(
                save_dir,
                f"{dataset}_{backbone}/{args.epochs}_{args.batch_size}_{args.optimizer}_{args.lr}/"
                f"{args.coeff_l_a_CE}_{args.coeff_l_cls_CE}_{args.coeff_prior_anch}/"
                f"{str(args.start_mse)}_{str(args.end_mse)}_{str(args.ema_momentum)}/"
                f"fold{fold_id}"
            )
            os.makedirs(run_dir, exist_ok=True)

            debug_dir = os.path.join(run_dir, "debug_preprocessing")
            os.makedirs(debug_dir, exist_ok=True)
            np.save(os.path.join(debug_dir, "X_train_pre_first8.npy"), locals()["X_train"][:8].astype(np.float32))
            np.save(os.path.join(debug_dir, "y_train_first8.npy"), np.asarray(y_train).reshape(-1)[:8])
            if a_train is not None:
                np.save(os.path.join(debug_dir, "a_train_first8.npy"), np.asarray(a_train).astype(np.float32)[:8])

            class_count_np = np.bincount(y_train.astype(np.int32), minlength=K).astype(np.float32)
            class_count = tf.convert_to_tensor(class_count_np, dtype=tf.float32)

            final_weights = os.path.join(run_dir, "final.weights.h5")
            if os.path.exists(final_weights):
                print(f"[SKIP] Run {fold_id} already completed")
                continue

            test_acc, _ = train_one_run(
                locals()["X_train"],
                y_train,
                a_train,
                locals()["X_val"],
                y_val,
                a_val,
                locals()["X_test"],
                y_test,
                a_test,
                M,
                K,
                backbone,
                args.img_size,
                args.lr,
                args.batch_size,
                args.epochs,
                args.coeff_l_a_CE,
                args.coeff_l_cls_CE,
                args.coeff_prior_anch,
                args.dataset,
                args.train_backbone,
                backbone_weights,
                run_dir,
                class_count,
                use_ema_prior=args.use_ema_prior,
                ema_momentum=args.ema_momentum,
                start_mse=args.start_mse,
                end_mse=args.end_mse,
                optimizer_name=args.optimizer,
                seed=run_seed,
            )
            print(f"[Run {fold_id}] Test balanced accuracy: {test_acc:.4f}")

            del class_count_np, class_count
            tf.keras.backend.clear_session()
            gc.collect()

if __name__ == "__main__":
    main()