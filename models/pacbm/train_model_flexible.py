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

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from data_loaders.data_loader import get_dataset_loaders
from data_loaders.configs import dataset_config
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, Callback

from models.pacbm.my_callbacks import DynamicAnchorWarmup

from models.pacbm.my_callbacks_flexible import (
    ConfMatSaver,
    FixedAnchorAlpha,
    EpochAnchorSchedule,
)
from models.pacbm.model_pacbm import PACBModel
from models.pacbm.utils import make_tfds, get_backbone_preprocess, numpy_from_dl, set_seeds, write_setting

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

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism not fully enabled:", e)


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


def extract_pil_augs(torch_transform):
    if isinstance(torch_transform, transforms.Compose):
        pil_transforms = []
        for t in torch_transform.transforms:
            if isinstance(t, transforms.ToTensor):
                break
            if isinstance(t, transforms.Normalize):
                continue
            pil_transforms.append(t)
        return transforms.Compose(pil_transforms) if pil_transforms else None
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


def create_optimizer(optimizer_name, lr):
    if not isinstance(optimizer_name, str):
        raise ValueError(f"Optimizer name must be a string, got {type(optimizer_name)}")

    name = optimizer_name.lower()
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

    opt = tf.keras.optimizers.get(optimizer_name)
    if hasattr(opt, "learning_rate"):
        opt.learning_rate = lr
    return opt


def compute_P_true_global(a_train, y_train, K, eps=1e-4):
    a_train = np.asarray(a_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    one_hot = np.eye(K, dtype=np.float32)[y_train]
    sum_attr = a_train.T @ one_hot
    class_count = one_hot.sum(axis=0)
    class_count_safe = np.maximum(class_count, 1.0)
    P_true = sum_attr / class_count_safe[None, :]
    P_true = np.clip(P_true, eps, 1.0 - eps)
    return P_true, class_count


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

        gamma_init = logit(P_true_global).astype("float32")
        model.class_layer.gamma.assign(gamma_init)

    optimizer = create_optimizer(optimizer_name, lr)
    model.compile(optimizer=optimizer, run_eagerly=False, weighted_metrics=[])
    return model


def build_anchor_callback(args):
    if args.coeff_prior_anch <= 0:
        return None

    if args.anchor_mode == "fixed":
        if args.fixed_alpha is None:
            raise ValueError("anchor_mode='fixed' requires --fixed_alpha")
        return FixedAnchorAlpha(alpha=args.fixed_alpha, verbose=1)

    if args.anchor_mode != "dynamic":
        raise ValueError(f"Unsupported anchor_mode={args.anchor_mode}")

    if args.schedule_type == "metric":
        return DynamicAnchorWarmup(
            start_score=args.start_mse,
            end_score=args.end_mse,
            ema=args.schedule_metric_ema,
            verbose=1,
        )

    return EpochAnchorSchedule(
        total_epochs=args.epochs,
        schedule_type=args.schedule_type,
        start_alpha=args.schedule_start_alpha,
        end_alpha=args.schedule_end_alpha,
        step_epoch=args.step_epoch,
        verbose=1,
    )


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
    anchor_callback=None,
):
    preprocess_fn = get_backbone_preprocess(backbone)

    def _map_preprocess(xb, yb, ab):
        xb = tf.cast(xb, tf.float32)
        xb_max = tf.reduce_max(xb)
        xb = tf.cond(xb_max <= 1.5, lambda: xb * 255.0, lambda: xb)
        xb = preprocess_fn(xb)
        return xb, yb, ab

    y_train = np.asarray(y_train).reshape(-1).astype(np.int32)
    y_val = np.asarray(y_val).reshape(-1).astype(np.int32)
    y_test = np.asarray(y_test).reshape(-1).astype(np.int32)
    a_train = np.asarray(a_train).astype(np.float32)
    a_val = np.asarray(a_val).astype(np.float32)
    a_test = np.asarray(a_test).astype(np.float32)

    train_ds = make_tfds(X_train, y_train, a_train, batch_size, shuffle=True, cache=False, seed=seed, reshuffle_each_iteration=False)
    val_ds = make_tfds(X_val, y_val, a_val, batch_size, shuffle=False, cache=False, seed=seed, reshuffle_each_iteration=False)
    test_ds = make_tfds(X_test, y_test, a_test, batch_size, shuffle=False, cache=False, seed=seed, reshuffle_each_iteration=False)

    train_ds = train_ds.map(_map_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(_map_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(_map_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    model = build_and_compile(
        M, K, backbone, img_size, lr, dataset_name, train_backbone,
        coeff_l_a_CE, coeff_l_cls_CE, coeff_prior_anch, save_dir, class_count,
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
    ckpt_every = EveryNEpochsCheckpoint(
        save_dir=weights_dir,
        every_n=300,
        verbose=1,
    )

    csv_logger = CSVLogger(os.path.join(save_dir, "history.csv"), append=False)
    es = EarlyStopping(monitor="val_bal_acc", mode="max", patience=200, restore_best_weights=True)

    callbacks = [cm_saver, ckpt_best, ckpt_every, csv_logger, es]
    if anchor_callback is not None:
        callbacks.insert(0, anchor_callback)

    model.fit(train_ds, validation_data=val_ds, epochs=n_epochs, callbacks=callbacks, verbose=2)

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

    model.save_weights(os.path.join(save_dir, "final.weights.h5"))
    return float(bal_acc), {"accuracy": float(acc), "balanced_accuracy": float(bal_acc), "confusion_matrix": cm}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c_annot", type=str, choices=["AwA2", "aPY", "CUB"], default="aPY")
    parser.add_argument("--dataset", type=str, choices=["CIFAR100", "AwA2", "aPY", "CUB"], default="aPY")
    parser.add_argument("--model", type=str, choices=["resnet50", "mobilenetv2", "efficientnetb0", "inceptionv3"], default="mobilenetv2")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
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
    parser.add_argument("--save_dir", type=str, default="./trained_models/test/")
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--train_backbone", action="store_true", default=True)
    parser.add_argument("--no_train_backbone", dest="train_backbone", action="store_false")
    parser.add_argument("--weights", type=str, default="imagenet", choices=["imagenet", "None"])

    # New flexibility-only controls.
    parser.add_argument("--anchor_mode", type=str, choices=["dynamic", "fixed"], default="dynamic")
    parser.add_argument("--fixed_alpha", type=float, default=None)
    parser.add_argument("--schedule_type", type=str, choices=["metric", "linear", "cosine", "step"], default="metric")
    parser.add_argument("--schedule_metric_ema", type=float, default=0.9)
    parser.add_argument("--schedule_start_alpha", type=float, default=0.0)
    parser.add_argument("--schedule_end_alpha", type=float, default=1.0)
    parser.add_argument("--step_epoch", type=int, default=None)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--only_fold", type=int, default=None, help="If set, run only this 1-based fold/run.")

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
            images, _, labels, _, annotations, _, mat_pd, _, concepts,
            label_to_idx_awa, mat_GT, train_tf, test_tf,
        ) = get_dataset_loaders(data_name, data_config, seed, save_dir, args.batch_size, args.data_dir)

        labels = np.asarray(labels).reshape(-1).astype(int)
        annotations = np.asarray(annotations).astype(np.float32)
        images = np.asarray(images)

        if float(np.max(images)) <= 1.01:
            base_images = (images * 255.0).astype("uint8")
        else:
            base_images = images.astype("uint8")

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

            X_train = X_train_aug.astype(np.uint8)
            X_val = X_val_aug.astype(np.uint8)
            X_test = X_test_aug.astype(np.uint8)

            tag_prefix = f"{args.run_tag}/" if args.run_tag else ""
            run_dir = os.path.join(
                save_dir,
                f"{tag_prefix}{dataset}_{backbone}/{args.epochs}_{args.batch_size}_{args.optimizer}_{args.lr}/"
                f"{args.coeff_l_a_CE}_{args.coeff_l_cls_CE}_{args.coeff_prior_anch}/fold{fold_id}",
            )
            os.makedirs(run_dir, exist_ok=True)

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

            #best_ckpt = os.path.join(run_dir, "models", f"epoch_{args.epochs}.weights.h5")
            best_ckpt = os.path.join(run_dir, "final.weights.h5")
            if os.path.exists(best_ckpt):
                print(f"[SKIP] Fold {fold_id} already completed")
                continue

            anchor_callback = build_anchor_callback(args)

            test_acc, _ = train_one_run(
                X_train, y_train, a_train,
                X_val, y_val, a_val,
                X_test, y_test, a_test,
                M, K, backbone, args.img_size, args.lr, args.batch_size, args.epochs,
                args.coeff_l_a_CE, args.coeff_l_cls_CE, args.coeff_prior_anch,
                args.dataset, args.train_backbone, backbone_weights, run_dir, class_count,
                P_true_global=P_true_global,
                use_ema_prior=args.use_ema_prior,
                ema_momentum=args.ema_momentum,
                start_mse=args.start_mse,
                end_mse=args.end_mse,
                optimizer_name=args.optimizer,
                seed=seed,
                anchor_callback=anchor_callback,
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
    elif dataset == "CUB":
        data_config = dataset_config["CUB"]
        result = get_dataset_loaders("CUB", data_config, seed, save_dir, args.batch_size, args.data_dir)

        train_dl, val_dl, test_dl, _, concepts, classes, concept_map, label_to_idx_awa = result

        X_train, y_train, a_train = numpy_from_dl(train_dl)
        X_val, y_val, a_val = numpy_from_dl(val_dl)
        X_test, y_test, a_test = numpy_from_dl(test_dl)

        for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
            arr = np.asarray(arr)
            assert arr.ndim == 4 and arr.shape[-1] in (1, 3), f"{name} shape {arr.shape} unexpected"
            assert np.isfinite(arr.astype(np.float32)).all(), f"{name} contains NaN/Inf"

        # Keep CUB arrays in memory once, then reuse them for repeated runs.
        # If the loader returns images in [0,1], convert them to [0,255]; train_one_run
        # applies the backbone-specific preprocessing batch-wise.
        if float(np.max(X_train)) <= 1.01:
            X_train = (X_train * 255.0).astype("float32")
            X_val = (X_val * 255.0).astype("float32")
            X_test = (X_test * 255.0).astype("float32")
        else:
            X_train = X_train.astype("float32")
            X_val = X_val.astype("float32")
            X_test = X_test.astype("float32")

        y_train = np.asarray(y_train).reshape(-1).astype(np.int32)
        y_val = np.asarray(y_val).reshape(-1).astype(np.int32)
        y_test = np.asarray(y_test).reshape(-1).astype(np.int32)
        a_train = np.asarray(a_train).astype(np.float32)
        a_val = np.asarray(a_val).astype(np.float32)
        a_test = np.asarray(a_test).astype(np.float32)

        M = a_train.shape[1] if a_train is not None else len(concepts)
        K = len(classes)

        tag_prefix = f"{args.run_tag}/" if args.run_tag else ""

        for fold_id in range(1, args.runs + 1):
            if args.only_fold is not None and fold_id != args.only_fold:
                continue

            run_seed = seed + fold_id - 1
            set_seeds(run_seed)
            torch.manual_seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)

            run_dir = os.path.join(
                save_dir,
                f"{tag_prefix}{dataset}_{backbone}/{args.epochs}_{args.batch_size}_{args.optimizer}_{args.lr}/"
                f"{args.coeff_l_a_CE}_{args.coeff_l_cls_CE}_{args.coeff_prior_anch}/fold{fold_id}",
            )
            os.makedirs(run_dir, exist_ok=True)

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

            final_weights = os.path.join(run_dir, "final.weights.h5")
            if os.path.exists(final_weights):
                print(f"[SKIP] Run {fold_id} already completed")
                continue

            class_count_np = np.bincount(y_train.astype(np.int32), minlength=K).astype(np.float32)
            class_count = tf.convert_to_tensor(class_count_np, dtype=tf.float32)
            P_true_global, _ = compute_P_true_global(a_train, y_train, K)

            # Build a fresh callback for every run. Reusing the same callback object
            # across folds/runs can carry over internal EMA state.
            anchor_callback = build_anchor_callback(args)

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
                P_true_global=None,
                use_ema_prior=args.use_ema_prior,
                ema_momentum=args.ema_momentum,
                start_mse=args.start_mse,
                end_mse=args.end_mse,
                optimizer_name=args.optimizer,
                seed=run_seed,
                anchor_callback=anchor_callback,
            )
            print(f"[Run {fold_id}] Test balanced accuracy: {test_acc:.4f}")

            del class_count_np, class_count, P_true_global, anchor_callback
            tf.keras.backend.clear_session()
            gc.collect()

    else:
        raise NotImplementedError(f"Dataset {dataset} is not supported by the PACBM ablation trainer.")


if __name__ == "__main__":
    main()
