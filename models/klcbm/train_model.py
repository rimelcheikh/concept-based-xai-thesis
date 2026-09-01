import os
import gc
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, Callback 
from collections import deque


from torchvision import transforms
from PIL import Image

from data_loaders.data_loader import get_dataset_loaders
from data_loaders.configs import dataset_config
from models.klcbm.utils import set_seeds, write_setting, numpy_from_dl

AUTOTUNE = tf.data.AUTOTUNE


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

PER_BACKBONE = {
    "mobilenetv2": ("sgd", 1e-2),
    "resnet50": ("adam", 2e-4),
    "efficientnetb0": ("adam", 1e-4),
    "inceptionv3": ("adam", 1e-4),
}

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



class StopIfRecentValBalAccTooLow(Callback):
    def __init__(self, threshold=0.1, window=10, monitor="val_bal_acc", min_epochs=None, verbose=1):
        super().__init__()
        self.threshold = threshold
        self.window = window
        self.monitor = monitor
        self.min_epochs = window if min_epochs is None else min_epochs
        self.verbose = verbose
        self.recent = deque(maxlen=window)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        epoch_num = epoch + 1

        if current is None:
            if self.verbose:
                print(f"\n[StopIfRecentValBalAccTooLow] '{self.monitor}' not found in logs.")
            return

        current = float(current)
        self.recent.append(current)

        if epoch_num >= self.min_epochs and len(self.recent) == self.window:
            if all(x < self.threshold for x in self.recent):
                if self.verbose:
                    vals = ", ".join(f"{x:.4f}" for x in self.recent)
                    print(
                        f"\n[Early Stop] Last {self.window} values of {self.monitor} "
                        f"are all below {self.threshold:.4f}: [{vals}]. Stopping."
                    )
                self.model.stop_training = True


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
    # default: identity
    return lambda x: x


def to_255_float(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    if x.size > 0 and np.nanmax(x) <= 1.5:
        x = x * 255.0
    return x


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
# Balanced Accuracy helper
# -------------------------
def balanced_accuracy_from_logits_or_probs(y_true_np, y_prob_np, K: int) -> float:
    y_true_np = y_true_np.reshape(-1).astype(np.int32)
    y_pred_np = np.argmax(y_prob_np, axis=1).astype(np.int32)

    cm = tf.math.confusion_matrix(y_true_np, y_pred_np, num_classes=K, dtype=tf.float32).numpy()
    tp = np.diag(cm)
    row_sum = cm.sum(axis=1)
    recall_c = np.divide(tp, row_sum, out=np.zeros_like(tp, dtype=float), where=row_sum > 0)
    mask = row_sum > 0
    return float(recall_c[mask].mean() if mask.any() else 0.0)


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
        y_oh = tf.one_hot(y_true, depth=K, dtype=tf.float32)   # smoothing handled inside CE
        per_sample = ce(y_oh, y_pred)                          # [B]
        w = tf.gather(cw, y_true)                              # [B]
        return tf.reduce_mean(per_sample * w)

    return loss_fn


# -------------------------
# Build backbone
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


# -------------------------
# KL-CBM Model (FIXED call/training)
# -------------------------
class AttributeClassProbability(tf.keras.layers.Layer):
    def __init__(self, K, M, class_count, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.M = M
        self.class_count = tf.cast(class_count, tf.float32)
        self.trainable = False

    def call(self, inputs):
        concatenated_attributes, labels = inputs
        labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)              # (B,)
        labels_oh = tf.one_hot(labels, depth=self.K, dtype=tf.float32)    # (B,K)
        labels_oh = tf.expand_dims(labels_oh, axis=1)                     # (B,1,K)

        a = tf.expand_dims(concatenated_attributes, axis=-1)              # (B,M,1)
        attribute_class_probs = a * labels_oh                             # (B,M,K)
        P_A_mk = tf.reduce_sum(attribute_class_probs, axis=0)             # (M,K)

        P_A_mk = tf.math.divide(P_A_mk, self.class_count)                 # normalize by counts
        P_A_mk.set_shape((self.M, self.K))
        return P_A_mk


class ProdLayer(tf.keras.layers.Layer):
    def __init__(self, K, M, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.M = M

    def call(self, inputs):
        attr_output, sum_output, labels = inputs

        labels_int = tf.cast(tf.reshape(labels, [-1]), tf.int32)
        counts = tf.cast(tf.math.bincount(labels_int, minlength=self.K), tf.float32)
        class_prob = counts / tf.reduce_sum(counts)

        labels_oh = tf.one_hot(labels_int, self.K)

        top_k_indices = tf.argsort(attr_output, axis=-1)[..., -5:]   # (B,5)
        attr_top_k = tf.experimental.numpy.take_along_axis(attr_output, top_k_indices, axis=-1)

        sum_top_k = tf.gather(sum_output, top_k_indices, axis=0)     # (B,5,K)

        div_output = attr_top_k[..., tf.newaxis] / tf.maximum(sum_top_k, tf.keras.backend.epsilon())  # (B,5,K)

        product_output = tf.reduce_prod(div_output, axis=1)           # (B,K)
        product_output *= class_prob                                  # (K,) broadcast
        product_output = product_output * labels_oh                   # keep only true-class 

        product_output.set_shape((None, self.K))
        return div_output, product_output, tf.nn.softmax(product_output, axis=-1)


class KLCBMmodel(tf.keras.Model):
    def __init__(self, img_size, M, K, backbone, coeff_l_attr, coeff_l_p_y, class_count):
        super().__init__()
        self.M = M
        self.K = K

        self.backbone_model = backbone
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()

        self.attr_layers = [
            tf.keras.layers.Dense(1, activation="sigmoid", name=f"attribute_output_{i}", use_bias=True)
            for i in range(M)
        ]
        self.conc_attr_layer = tf.keras.layers.Concatenate(name="concatenated_attributes")

        self.sum_layer = AttributeClassProbability(K, M, class_count, name="sum_layer")
        self.prod_layer = ProdLayer(K, M, name="prod_layer")

        self.final_layer = tf.keras.layers.Dense(K, activation="softmax", name="final_output", use_bias=False, trainable=True)

        self.MSE = tf.keras.losses.MeanSquaredError()
        self.KLD = tf.keras.losses.KLDivergence(reduction="sum_over_batch_size", name="kl_divergence")

        self.coeff_l_attr = coeff_l_attr
        self.coeff_l_p_y = coeff_l_p_y

        self.metric_attr_mse = tf.keras.metrics.MeanSquaredError(name="attr_mse")
        self.metric_final_acc = tf.keras.metrics.SparseCategoricalAccuracy(name="final_acc")

    def call(self, inputs, training=None):
        images, labels = inputs  # labels are used inside sum/prod layers

        x = self.backbone_model(images, training=training)
        x = self.global_avg_pool(x)

        attr_outs = [layer(x) for layer in self.attr_layers]                # list of (B,1)
        concatenated_attributes = self.conc_attr_layer(attr_outs)           # (B,M)

        sum_output = self.sum_layer([concatenated_attributes, labels])      # (M,K)
        div_output, prod_output, soft_prod_output = self.prod_layer([concatenated_attributes, sum_output, labels])

        final_output = self.final_layer(concatenated_attributes)            # (B,K)
        final_output.set_shape((None, self.K))
        return concatenated_attributes, sum_output, div_output, prod_output, soft_prod_output, final_output

    def train_step(self, data):
        (images, in_class_labels), (ann_input, class_labels) = data
        class_labels = tf.cast(tf.reshape(class_labels, [-1]), tf.int32)
        one_hot = tf.one_hot(class_labels, depth=self.K)

        with tf.GradientTape() as tape:
            attr_pred, sum_output, div_output, prod_output, soft_prod_output, y_pred = \
                self([images, in_class_labels], training=True)

            loss_attr = self.MSE(ann_input, attr_pred)
            loss_p_y = self.KLD(tf.stop_gradient(soft_prod_output), y_pred)
            total_loss = self.coeff_l_attr * loss_attr + self.coeff_l_p_y * loss_p_y

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.metric_attr_mse.update_state(ann_input, attr_pred)
        self.metric_final_acc.update_state(class_labels, y_pred)

        return {
            "loss": total_loss,
            "attr_loss": loss_attr,
            "kld_loss": loss_p_y,
            "attr_mse": self.metric_attr_mse.result(),
            "final_acc": self.metric_final_acc.result(),
        }

    def test_step(self, data):
        (images, in_class_labels), (ann_input, class_labels) = data
        class_labels = tf.cast(tf.reshape(class_labels, [-1]), tf.int32)

        attr_pred, sum_output, div_output, prod_output, soft_prod_output, y_pred = \
            self([images, in_class_labels], training=False)

        self.metric_attr_mse.update_state(ann_input, attr_pred)
        self.metric_final_acc.update_state(class_labels, y_pred)

        return {
            "attr_mse": self.metric_attr_mse.result(),
            "final_acc": self.metric_final_acc.result(),
        }


# -------------------------
# tf.data builder for this model
# -------------------------
def make_ds_for_klcbm(X, y, a, batch_size, shuffle, seed, preprocess_fn):
    y = y.reshape(-1).astype("int32")
    a = a.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices(((X, y), (a, y)))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(y), seed=seed, reshuffle_each_iteration=False)

    def _map_preprocess(x_pair, y_pair):
        xb, y_in = x_pair
        ab, y_true = y_pair

        xb = tf.cast(xb, tf.float32)
        xb_max = tf.reduce_max(xb)
        xb = tf.cond(xb_max <= 1.5, lambda: xb * 255.0, lambda: xb)
        xb = preprocess_fn(xb)

        return (xb, y_in), (ab, y_true)

    ds = ds.batch(batch_size).map(_map_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return ds


# -------------------------
# Train+Eval one fold
# -------------------------
def train_one_fold_klcbm(
    X_train, y_train, a_train,
    X_val, y_val, a_val,
    X_test, y_test, a_test,
    backbone_name, img_size,
    batch_size, n_epochs,
    lr, optimizer_name,
    coeff_l_attr, coeff_l_p_y,
    backbone_weights,
    train_backbone,
    save_dir,
    seed,
):
    preprocess_fn = get_backbone_preprocess(backbone_name)

    K = int(np.max(y_train)) + 1
    class_count = tf.math.bincount(tf.constant(y_train, dtype=tf.int32), minlength=K, maxlength=K, dtype=tf.float32)

    train_ds = make_ds_for_klcbm(X_train, y_train, a_train, batch_size, shuffle=True,  seed=seed, preprocess_fn=preprocess_fn)
    val_ds   = make_ds_for_klcbm(X_val,   y_val,   a_val,   batch_size, shuffle=False, seed=seed, preprocess_fn=preprocess_fn)
    test_ds  = make_ds_for_klcbm(X_test,  y_test,  a_test,  batch_size, shuffle=False, seed=seed, preprocess_fn=preprocess_fn)

    input_shape = (img_size, img_size, 3)
    backbone = build_backbone(backbone_name, input_shape, weights=backbone_weights, train_backbone=train_backbone)

    model = KLCBMmodel(
        img_size=img_size,
        M=a_train.shape[1],
        K=K,
        backbone=backbone,
        coeff_l_attr=coeff_l_attr,
        coeff_l_p_y=coeff_l_p_y,
        class_count=class_count,
    )

    if optimizer_name.lower() == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=1.0)
    elif optimizer_name.lower() == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=lr, clipvalue=1.0, momentum=0.9)
    else:
        opt = tf.keras.optimizers.get(optimizer_name)
        if hasattr(opt, "learning_rate"):
            opt.learning_rate = lr

    model.compile(
        optimizer=opt,
        jit_compile=False,
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
    ckpt_every_10 = EveryNEpochsCheckpoint(
        save_dir=weights_dir,
        every_n=300,
        verbose=1,
    )

    csv_logger = CSVLogger(os.path.join(save_dir, "history.csv"), append=False)
    es = EarlyStopping(
        monitor="val_bal_acc",
        mode="max",
        patience=200,
        restore_best_weights=True,
    )

    stop_if_low_val_bal_acc = StopIfRecentValBalAccTooLow(
        threshold=0.1,
        window=10,
        monitor="val_bal_acc",
        min_epochs=30,
        verbose=1,
    )

    val_bal_cb = FullValBalancedAcc(
        val_ds=val_ds,
        K=K,
        get_probs_fn=lambda model, batch: (
            batch[1][1].numpy(),  # y_true
            model([batch[0][0], batch[0][1]], training=False)[-1].numpy(),  # probs
        ),
    )
    #cm_saver = ConfMatSaver(save_dir=save_dir, K=K, normalize_rows=True, save_every=1)

    callbacks = [
        val_bal_cb,
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

    if os.path.exists(best_path):
        model.load_weights(best_path)
    else:
        print("WARNING: best checkpoint not found:", best_path)

    # Manual TEST balanced accuracy
    y_true_all, y_prob_all = [], []
    for (xb, yb_in), (ab, yb_true) in test_ds:
        *_, y_prob = model([xb, yb_in], training=False)
        y_true_all.append(yb_true.numpy())
        y_prob_all.append(y_prob.numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    test_bal = balanced_accuracy_from_logits_or_probs(y_true, y_prob, K)

    model.save_weights(os.path.join(save_dir, "final.weights.h5"))
    return test_bal


def run_all(args):
    seed = 42
    set_seeds(seed)

    dataset = args.dataset
    backbone = args.model

    backbone_weights = None if args.weights == "None" else "imagenet"

    data_cfg = dataset_config.get(dataset)

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
            data_name, data_cfg, seed, args.save_dir,
            args.batch_size, args.data_dir
        )

        if annotations is None:
            raise ValueError(f"{dataset}: annotations is None but KL-CBM needs concept supervision.")

        if images.max() <= 1.01:
            base_images = (images * 255.0).astype("uint8")
        else:
            base_images = images.astype("uint8")

        labels = np.asarray(labels).reshape(-1).astype(np.int32)
        annotations = np.asarray(annotations).astype(np.float32)

        train_pil_aug = extract_pil_augs(train_tf)
        test_pil_aug  = extract_pil_augs(test_tf)

        kf = StratifiedKFold(n_splits=args.runs, shuffle=True, random_state=seed)

    elif dataset == "CUB":
        data_name = "CUB"
        result = get_dataset_loaders(
            data_name, data_cfg, seed, args.save_dir,
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

        y_train_full = np.asarray(y_train_full).reshape(-1).astype(np.int32)
        y_val_full = np.asarray(y_val_full).reshape(-1).astype(np.int32)
        y_test_full = np.asarray(y_test_full).reshape(-1).astype(np.int32)

        a_train_full = np.asarray(a_train_full).astype(np.float32)
        a_val_full = np.asarray(a_val_full).astype(np.float32)
        a_test_full = np.asarray(a_test_full).astype(np.float32)

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
    print(f"KL-CBM | dataset={dataset} | backbone={backbone} | opt={optimizer_name} | lr={lr} | img_size={img_size}")
    print("==============================\n")

    if dataset in ["AwA2", "aPY"]:
        fold_id = 0
        for full_train_idx, test_idx in kf.split(base_images, labels):
            fold_id += 1
            if args.only_fold is not None and fold_id != args.only_fold:
                continue

            run_dir = os.path.join(
                args.save_dir,
                f"{dataset}_{backbone}/{args.epochs}_{args.batch_size}_{optimizer_name}_{lr}/"
                f"{args.coeff_attr}_{args.coeff_py}/fold{fold_id}"
            )
            os.makedirs(run_dir, exist_ok=True)

            done_flag = os.path.join(run_dir, "final.weights.h5")
            if os.path.exists(done_flag):
                print(f"[SKIP] {dataset} | {backbone} | fold {fold_id} already done")
                continue

            tr_idx, val_idx = train_test_split(
                full_train_idx,
                test_size=args.val_split,
                random_state=seed,
                stratify=labels[full_train_idx],
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

            K = int(np.max(labels)) + 1
            M = int(annotations.shape[1])

            write_setting(
                run_dir, backbone, dataset, dataset,
                args.coeff_attr, None, args.coeff_py,
                fold_id, args.runs, optimizer_name, lr, args.batch_size,
                None, None,
                args.val_split, args.epochs, len(y_train), len(y_val), len(y_test), M, K
            )

            bal = train_one_fold_klcbm(
                X_train, y_train, a_train,
                X_val, y_val, a_val,
                X_test, y_test, a_test,
                backbone_name=backbone,
                img_size=img_size,
                batch_size=args.batch_size,
                n_epochs=args.epochs,
                lr=lr,
                optimizer_name=optimizer_name,
                coeff_l_attr=args.coeff_attr,
                coeff_l_p_y=args.coeff_py,
                backbone_weights=backbone_weights,
                train_backbone=args.train_backbone,
                save_dir=run_dir,
                seed=seed,
            )

            print(f"[DONE] {dataset} | {backbone} | fold {fold_id} | test bal acc: {bal:.4f}")

            tf.keras.backend.clear_session()
            gc.collect()

    else:
        for fold_id in range(1, args.runs + 1):
            if args.only_fold is not None and fold_id != args.only_fold:
                continue

            run_dir = os.path.join(
                args.save_dir,
                f"{dataset}_{backbone}/{args.epochs}_{args.batch_size}_{optimizer_name}_{lr}/"
                f"{args.coeff_attr}_{args.coeff_py}/fold{fold_id}"
            )
            os.makedirs(run_dir, exist_ok=True)

            done_flag = os.path.join(run_dir, "final.weights.h5")
            if os.path.exists(done_flag):
                print(f"[SKIP] {dataset} | {backbone} | fold {fold_id} already done")
                continue

            run_seed = seed + fold_id - 1
            set_seeds(run_seed)

            X_train = apply_pil_augs_np(X_train_full, train_pil_aug, img_size).astype(np.uint8)
            X_val   = apply_pil_augs_np(X_val_full,   test_pil_aug,  img_size).astype(np.uint8)
            X_test  = apply_pil_augs_np(X_test_full,  test_pil_aug,  img_size).astype(np.uint8)

            y_train = y_train_full
            y_val   = y_val_full
            y_test  = y_test_full

            a_train = a_train_full
            a_val   = a_val_full
            a_test  = a_test_full

            K = len(classes)
            M = int(a_train.shape[1])

            write_setting(
                run_dir, backbone, dataset, dataset,
                args.coeff_attr, None, args.coeff_py,
                fold_id, args.runs, optimizer_name, lr, args.batch_size,
                None, None,
                None, args.epochs, len(y_train), len(y_val), len(y_test), M, K
            )

            bal = train_one_fold_klcbm(
                X_train, y_train, a_train,
                X_val, y_val, a_val,
                X_test, y_test, a_test,
                backbone_name=backbone,
                img_size=img_size,
                batch_size=args.batch_size,
                n_epochs=args.epochs,
                lr=lr,
                optimizer_name=optimizer_name,
                coeff_l_attr=args.coeff_attr,
                coeff_l_p_y=args.coeff_py,
                backbone_weights=backbone_weights,
                train_backbone=args.train_backbone,
                save_dir=run_dir,
                seed=run_seed,
            )

            print(f"[DONE] {dataset} | {backbone} | fold {fold_id} | test bal acc: {bal:.4f}")

            tf.keras.backend.clear_session()
            gc.collect()


def main():
    parser = argparse.ArgumentParser()

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
    parser.add_argument("--val_split", type=float, default=0.2)

    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--override_hparams", action="store_true", default=False)

    parser.add_argument("--coeff_attr", type=float, default=5.0)
    parser.add_argument("--coeff_py", type=float, default=5.0)

    parser.add_argument("--save_dir", type=str, default="./trained_models/klcbm/")
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
