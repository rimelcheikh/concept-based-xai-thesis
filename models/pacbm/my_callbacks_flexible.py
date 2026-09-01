import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# ---------------------------------------------------------
# Confusion-matrix metrics helper
# ---------------------------------------------------------
def cm_to_metrics(cm: np.ndarray):
    """Compute balanced acc, macro precision/recall/F1 from confusion matrix."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)

    recall_c = np.divide(tp, row_sum, out=np.zeros_like(tp), where=row_sum > 0)
    prec_c = np.divide(tp, col_sum, out=np.zeros_like(tp), where=col_sum > 0)

    mask_true = row_sum > 0
    mask_pred = col_sum > 0
    mask_f1 = mask_true & ((prec_c + recall_c) > 0)

    bal_acc = recall_c[mask_true].mean() if mask_true.any() else 0.0
    macro_recall = recall_c[mask_true].mean() if mask_true.any() else 0.0
    macro_precision = prec_c[mask_pred].mean() if mask_pred.any() else 0.0

    f1_c = np.divide(
        2 * prec_c * recall_c,
        (prec_c + recall_c),
        out=np.zeros_like(tp),
        where=(prec_c + recall_c) > 0,
    )
    macro_f1 = f1_c[mask_f1].mean() if mask_f1.any() else 0.0

    return {
        "balanced_accuracy": float(bal_acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "support": int(row_sum.sum()),
    }


class ConfusionMatrixMetrics(tf.keras.metrics.Metric):
    """
    Accumulates a KxK confusion matrix over an epoch and exposes:
      - accuracy
      - balanced_accuracy (macro recall)
      - macro_precision
      - macro_recall
      - macro_f1
    """

    def __init__(self, num_classes: int, name="cm_metrics", **kwargs):
        super().__init__(name=name, **kwargs)
        self.K = int(num_classes)
        self.cm = self.add_weight(
            name="cm",
            shape=(self.K, self.K),
            initializer="zeros",
            dtype=tf.float32,
        )

    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.int32)
        cm_batch = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.K, dtype=tf.float32)
        self.cm.assign_add(cm_batch)

    def result(self):
        eps = 1e-8
        cm = self.cm

        tp = tf.linalg.diag_part(cm)
        row_sum = tf.reduce_sum(cm, axis=1)
        col_sum = tf.reduce_sum(cm, axis=0)

        recall_c = tf.where(row_sum > 0, tp / (row_sum + eps), tf.zeros_like(tp))
        prec_c = tf.where(col_sum > 0, tp / (col_sum + eps), tf.zeros_like(tp))

        acc = tf.reduce_sum(tp) / (tf.reduce_sum(cm) + eps)

        mask_true = row_sum > 0
        bal_acc = tf.cond(
            tf.reduce_any(mask_true),
            lambda: tf.reduce_mean(tf.boolean_mask(recall_c, mask_true)),
            lambda: tf.constant(0.0, tf.float32),
        )

        mask_pred = col_sum > 0
        macro_prec = tf.cond(
            tf.reduce_any(mask_pred),
            lambda: tf.reduce_mean(tf.boolean_mask(prec_c, mask_pred)),
            lambda: tf.constant(0.0, tf.float32),
        )

        macro_rec = bal_acc

        f1_c = tf.where(
            (prec_c + recall_c) > 0,
            2.0 * prec_c * recall_c / (prec_c + recall_c + eps),
            tf.zeros_like(tp),
        )
        mask_f1 = mask_true & ((prec_c + recall_c) > 0)
        macro_f1 = tf.cond(
            tf.reduce_any(mask_f1),
            lambda: tf.reduce_mean(tf.boolean_mask(f1_c, mask_f1)),
            lambda: tf.constant(0.0, tf.float32),
        )

        return {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
        }


class ConfMatSaver(tf.keras.callbacks.Callback):
    """
    Uses model.cm_train and model.cm_val (KxK tensors) and saves:
      - .npy and .png matrices
      - metrics_per_epoch.csv
    """

    def __init__(self, save_dir, K, normalize_rows=True, save_every=1, **kwargs):
        super().__init__(**kwargs)
        self.save_dir = save_dir
        self.K = K
        self.normalize = normalize_rows
        self.save_every = save_every

        self.cm_root = os.path.join(self.save_dir, "cm_figs")
        self.cm_train_dir = os.path.join(self.cm_root, "train")
        self.cm_val_dir = os.path.join(self.cm_root, "val")
        os.makedirs(self.cm_train_dir, exist_ok=True)
        os.makedirs(self.cm_val_dir, exist_ok=True)

        self.metrics_csv = os.path.join(self.save_dir, "metrics_per_epoch.csv")
        if not os.path.exists(self.metrics_csv):
            with open(self.metrics_csv, "w", encoding="utf-8") as f:
                f.write(
                    "epoch,split,balanced_accuracy,macro_precision,"
                    "macro_recall,macro_f1,support\n"
                )

    def _save_cm(self, cm, split_name, epoch):
        out_dir = self.cm_train_dir if split_name == "train" else self.cm_val_dir
        np.save(os.path.join(out_dir, f"cm_{split_name}_epoch_{epoch}.npy"), cm)

        cm_plot = cm.astype(np.float32)
        if self.normalize:
            row_sums = cm_plot.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_plot = cm_plot / row_sums

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm_plot, interpolation="nearest", aspect="auto")
        ax.set_title(f"Confusion Matrix ({split_name}) — epoch {epoch}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.colorbar(im, fraction=0.046, pad=0.04)

        tick_step = max(1, self.K // 10)
        ticks = np.arange(0, self.K, tick_step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks)
        ax.set_yticklabels(ticks)

        if self.K <= 2:
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=30)

        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cm_{split_name}_epoch_{epoch}.png"), dpi=150)
        plt.close(fig)

    def on_train_begin(self, logs=None):
        self.model.reset_epoch_confusions()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_every != 0:
            self.model.reset_epoch_confusions()
            return

        cm_train = self.model.cm_train.numpy()
        cm_val = self.model.cm_val.numpy()

        self._save_cm(cm_train, "train", epoch + 1)
        self._save_cm(cm_val, "val", epoch + 1)

        m_train = cm_to_metrics(cm_train)
        m_val = cm_to_metrics(cm_val)

        with open(self.metrics_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch+1},train,{m_train['balanced_accuracy']:.6f},"
                f"{m_train['macro_precision']:.6f},"
                f"{m_train['macro_recall']:.6f},"
                f"{m_train['macro_f1']:.6f},{m_train['support']}\n"
            )
            f.write(
                f"{epoch+1},val,{m_val['balanced_accuracy']:.6f},"
                f"{m_val['macro_precision']:.6f},"
                f"{m_val['macro_recall']:.6f},"
                f"{m_val['macro_f1']:.6f},{m_val['support']}\n"
            )

        self.model.reset_epoch_confusions()


class DynamicAnchorWarmup(tf.keras.callbacks.Callback):
    """
    Sets model.anchor_alpha each epoch based on concept quality.

    Continuous concepts (AwA*): uses concept_mse_macro (lower is better)
    Binary concepts (aPY):     uses 1 - concept_f1_macro (lower is better),
                              falls back to concept_bce_macro if F1 not available.

    - If score >= start_score => alpha = 0 (TRUE priors)
    - If score <= end_score   => alpha = 1 (PRED priors)
    - Otherwise linearly interpolated.

    Uses EMA smoothing for stability.
    """

    def __init__(self, start_score=0.20, end_score=0.10, ema=0.9, verbose=1):
        super().__init__()
        self.start_score = float(start_score)
        self.end_score = float(end_score)
        self.ema = float(ema)
        self.verbose = int(verbose)
        self.smooth_score = None

    def on_train_begin(self, logs=None):
        self.smooth_score = self.start_score
        self.model.anchor_alpha.assign(0.0)

    def _get_concept_score(self, logs: dict):
        loss_name = getattr(self.model, "concept_loss_name", None)
        if loss_name == "bce":
            f1 = logs.get("val_concept_f1_macro", None)
            if f1 is not None:
                f1 = float(f1)
                return 1.0 - f1, "1 - val_concept_f1_macro"

            bce = logs.get("val_concept_bce_macro", logs.get("val_concept_bce_micro", logs.get("val_loss_a_BCE", None)))
            if bce is not None:
                return float(bce), "val_concept_bce_macro"
            return None, None

        mse = logs.get("val_concept_mse_macro", logs.get("val_concept_mse_micro", logs.get("val_loss_a_MSE", None)))
        if mse is not None:
            return float(mse), "val_concept_mse_macro"
        return None, None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        score, score_name = self._get_concept_score(logs)
        if score is None:
            return

        self.smooth_score = self.ema * self.smooth_score + (1.0 - self.ema) * score

        denom = max(1e-8, (self.start_score - self.end_score))
        alpha = (self.start_score - self.smooth_score) / denom
        alpha = float(np.clip(alpha, 0.0, 1.0))

        self.model.anchor_alpha.assign(alpha)

        if self.verbose:
            print(
                f"\n[DynamicAnchorWarmup] epoch {epoch+1:03d} | "
                f"score={score:.5f} ({score_name}) smooth={self.smooth_score:.5f} -> alpha={alpha:.3f}"
            )


class FixedAnchorAlpha(tf.keras.callbacks.Callback):
    """Keep anchor_alpha fixed for the entire run."""

    def __init__(self, alpha: float, verbose: int = 1):
        super().__init__()
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.verbose = int(verbose)

    def on_train_begin(self, logs=None):
        self.model.anchor_alpha.assign(self.alpha)
        if self.verbose:
            print(f"[FixedAnchorAlpha] alpha={self.alpha:.3f}")

    def on_epoch_begin(self, epoch, logs=None):
        self.model.anchor_alpha.assign(self.alpha)


class EpochAnchorSchedule(tf.keras.callbacks.Callback):
    """
    Epoch-based anchor schedule that only changes how anchor_alpha is exposed.
    It does not alter the model or losses.

    schedule_type:
      - linear: alpha increases linearly from start_alpha to end_alpha
      - cosine: cosine ramp from start_alpha to end_alpha
      - step:   alpha=start_alpha before step_epoch, else end_alpha
    """

    def __init__(
        self,
        total_epochs: int,
        schedule_type: str = "linear",
        start_alpha: float = 0.0,
        end_alpha: float = 1.0,
        step_epoch: int | None = None,
        verbose: int = 1,
    ):
        super().__init__()
        self.total_epochs = int(total_epochs)
        self.schedule_type = str(schedule_type).lower()
        self.start_alpha = float(start_alpha)
        self.end_alpha = float(end_alpha)
        self.step_epoch = None if step_epoch is None else int(step_epoch)
        self.verbose = int(verbose)

        if self.schedule_type not in {"linear", "cosine", "step"}:
            raise ValueError(f"Unsupported schedule_type={schedule_type}")
        if self.schedule_type == "step" and self.step_epoch is None:
            raise ValueError("step schedule requires step_epoch")

    def _alpha_for_epoch(self, epoch_index: int) -> float:
        if self.total_epochs <= 1:
            return self.end_alpha

        if self.schedule_type == "step":
            return self.start_alpha if epoch_index < self.step_epoch else self.end_alpha

        t = epoch_index / max(1, self.total_epochs - 1)
        if self.schedule_type == "linear":
            alpha = self.start_alpha + (self.end_alpha - self.start_alpha) * t
        elif self.schedule_type == "cosine":
            cosine_t = 0.5 * (1.0 - np.cos(np.pi * t))
            alpha = self.start_alpha + (self.end_alpha - self.start_alpha) * cosine_t
        else:
            raise ValueError(f"Unsupported schedule_type={self.schedule_type}")

        return float(np.clip(alpha, 0.0, 1.0))

    def on_train_begin(self, logs=None):
        alpha = self._alpha_for_epoch(0)
        self.model.anchor_alpha.assign(alpha)
        if self.verbose:
            print(f"[EpochAnchorSchedule] type={self.schedule_type} alpha(epoch=1)={alpha:.3f}")

    def on_epoch_begin(self, epoch, logs=None):
        alpha = self._alpha_for_epoch(epoch)
        self.model.anchor_alpha.assign(alpha)
        if self.verbose:
            print(f"[EpochAnchorSchedule] epoch {epoch+1:03d} -> alpha={alpha:.3f}")


class PAmkEMAvsRawCallback(tf.keras.callbacks.Callback):
    """
    Saves heatmaps of:
      - Raw batch P(A_m|k) from last train batch
      - EMA P(A_m|k) from ConceptClassPriorsLayer
    """

    def __init__(self, model, save_dir, K, M, log_every=1, **kwargs):
        super().__init__(**kwargs)
        self.save_dir = save_dir
        self.K = K
        self.M = M
        self.log_every = log_every
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_every != 0:
            return

        ema_matrix = self.model.concept_class_priors.P_pred_ema.numpy()
        raw_matrix = self._compute_PAmk_raw(self.model)

        self._plot_heatmap(raw_matrix, epoch, "Raw P(Aₘ|k)", "raw")
        self._plot_heatmap(ema_matrix, epoch, "EMA P(Aₘ|k)", "ema")

    def _compute_PAmk_raw(self, model):
        last_attr = model._last_batch_attr_preds
        last_labels = model._last_train_batch_labels

        labels = tf.cast(tf.reshape(last_labels, [-1]), tf.int32)
        labels_oh = tf.one_hot(labels, depth=self.K, dtype=tf.float32)
        labels_oh = tf.expand_dims(labels_oh, axis=1)

        concatenated_attributes = tf.expand_dims(last_attr, axis=-1)
        attribute_class_probs = concatenated_attributes * labels_oh
        P_A_mk_batch = tf.reduce_sum(attribute_class_probs, axis=0)

        class_count = tf.math.bincount(labels, minlength=self.K, maxlength=self.K, dtype=tf.float32)
        mask = class_count == 0
        safe_class_count = tf.where(mask, tf.ones_like(class_count), class_count)
        P_A_mk_batch = tf.math.divide(P_A_mk_batch, safe_class_count)
        P_A_mk_batch = tf.where(mask, tf.zeros_like(P_A_mk_batch), P_A_mk_batch)

        return P_A_mk_batch.numpy()

    def _plot_heatmap(self, matrix, epoch, title, suffix):
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.title(f"{title} at Epoch {epoch + 1}")
        plt.xlabel("Class")
        plt.ylabel("Concept (Feature)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"p_amk_{suffix}_epoch_{epoch + 1}.png"), dpi=150)
        plt.close()


class RunSummarySaver(tf.keras.callbacks.Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.best = {
            "val_bal_acc": 0.0,
            "val_concept_mse_macro": float("inf"),
            "val_loss_prior_anch": float("inf"),
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if "val_bal_acc_epoch" in logs:
            self.best["val_bal_acc"] = max(self.best["val_bal_acc"], logs["val_bal_acc_epoch"])

        if "val_concept_mse_macro" in logs:
            self.best["val_concept_mse_macro"] = min(self.best["val_concept_mse_macro"], logs["val_concept_mse_macro"])

        if "val_loss_prior_anch" in logs:
            self.best["val_loss_prior_anch"] = min(self.best["val_loss_prior_anch"], logs["val_loss_prior_anch"])

    def on_train_end(self, logs=None):
        import json

        with open(os.path.join(self.save_dir, "run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(self.best, f, indent=2)
