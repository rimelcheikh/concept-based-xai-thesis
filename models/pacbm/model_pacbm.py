import tensorflow as tf
from models.pacbm.my_callbacks import ConfusionMatrixMetrics


# ---------------------------------------------------------
# Priors layer P(A_m | y=k)
# ---------------------------------------------------------
class ConceptClassPriorsLayer(tf.keras.layers.Layer):
    """
    Computes:
      - P_pred_used: P(A_m|k) from predicted concepts and true labels,
                     either raw batch-driven or EMA-smoothed.
      - P_true_batch: batch-driven P(A_m|k) from ground-truth annotations.

    Returns (P_pred_used, P_true_batch), each [M, K].
    """

    def __init__(self, K, M, class_count, use_ema=False, momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.M = M
        self.class_count = tf.cast(class_count, tf.float32)  # [K]
        self.use_ema = bool(use_ema)
        self.momentum = float(momentum)

        # EMA over P_pred_batch
        self.P_pred_ema = self.add_weight(
            name="P_pred_ema",
            shape=(M, K),
            initializer=tf.keras.initializers.Constant(1e-3),
            trainable=False,
            dtype=tf.float32,
        )

    def call(self, inputs, training=None):
        """
        inputs: (concepts_pred, labels, attr_ann)
          - concepts_pred: [B, M]   predicted concept probabilities
          - labels:        [B]      integer class labels
          - attr_ann:      [B, M]   ground-truth concept annotations
        """
        concepts_pred, labels, attr_ann = inputs  # [B,M], [B], [B,M]

        labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)                 # [B]
        attr_ann = tf.cast(attr_ann, tf.float32)
        
        one_hot_labels = tf.one_hot(labels, depth=self.K, dtype=tf.float32)  # [B,K]
        batch_counts = tf.reduce_sum(one_hot_labels, axis=0)                 # [K]

        # --- TRUE priors from annotations: P_true_batch[m,k] = P(A_m=1 | y=k) in this batch ---
        P_true_batch = tf.transpose(attr_ann) @ one_hot_labels               # [M,K]
        P_true_batch = tf.math.divide_no_nan(
            P_true_batch, tf.maximum(batch_counts, 1.0)
        )  # [M,K]


        # --- PRED priors from predicted concepts ---
        labels_oh = tf.expand_dims(one_hot_labels, axis=1)  # [B,1,K]
        conc = tf.expand_dims(concepts_pred, axis=-1)       # [B,M,1]
        pred_class_probs = conc * labels_oh                 # [B,M,K]
        P_pred_batch = tf.reduce_sum(pred_class_probs, axis=0)  # [M,K]

        # normalize by how many examples of each class are in THIS batch
        P_pred_batch = tf.math.divide_no_nan(
            P_pred_batch,
            tf.maximum(batch_counts, 1.0)
        )  # [M,K], ~ P(A_m=1 | y=k) in batch

        if self.use_ema:
            # First call: if ema == const(1e-3) everywhere, just take the batch
            is_first = tf.reduce_all(tf.equal(self.P_pred_ema, 1e-3))
            new_ema = tf.cond(
                is_first,
                lambda: P_pred_batch,
                lambda: self.momentum * self.P_pred_ema
                        + (1.0 - self.momentum) * P_pred_batch,
            )
            self.P_pred_ema.assign(new_ema)
            P_pred_used = self.P_pred_ema
        else:
            P_pred_used = P_pred_batch

        return P_pred_used, P_true_batch


# ---------------------------------------------------------
# Transparent classifier over concepts
# ---------------------------------------------------------
class ClassificationLayer(tf.keras.layers.Layer):
    """
    Transparent concept-based classifier:

        log p(y = k | x) ∝ sum_m gamma_{m,k} * logit(a_m) + bias_k

    where a_m are concept probabilities.
    """

    def __init__(self, K, M, **kwargs):
        super().__init__(**kwargs)
        self.K, self.M = K, M

        self.gamma = self.add_weight(
            name="gamma_mk",
            shape=(M, K),
            initializer="glorot_uniform",
            trainable=True,
            regularizer=tf.keras.regularizers.l2(1e-4),  
        )
        self.bias = self.add_weight(
            name="bias_k",
            shape=(K,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, concepts_pred, training=None):
        """
        concepts_pred: [B,M] concept probabilities in (0,1)
        """
        eps = 1e-2
        a = tf.clip_by_value(concepts_pred, eps, 1.0 - eps)  # [B,M]
        logit_a = tf.math.log(a) - tf.math.log(1.0 - a)      # [B,M]

        logits = tf.linalg.matmul(logit_a, self.gamma) + self.bias  # [B,K]
        probs = tf.nn.softmax(logits, axis=-1)                       # [B,K]
        return logits, probs



class PACBModel(tf.keras.Model):
    """
    PACBM Model with dynamic TRUE->PRED prior anchoring.

    - Backbone: MobileNetV2 / ResNet50 / EfficientNetB0 (or custom keras.Model).
    - Concepts: M sigmoid heads from backbone features.
    - Classifier: log P(y|x) is linear in logit(concepts).
    - Priors: P(A_m|y=k) from ConceptClassPriorsLayer.
    - Dynamic anchoring:
        anchor_alpha = 0 -> anchor gamma to TRUE priors
        anchor_alpha = 1 -> anchor gamma to PRED priors
      updated by DynamicAnchorWarmup callback.
    """

    def __init__(
        self,
        input_size,      # (H, W, 3)
        M,
        K,
        backbone_name="mobilenetv2",  
        backbone_weights="imagenet",   
        dataset_name=None,
        train_backbone=True,
        coeff_l_a_CE=1.0,
        coeff_l_cls_CE=1.0,
        coeff_prior_anch=1.0,
        save_dir=None,
        class_count=None,
        use_ema_prior=False,
        ema_momentum=0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.M = M
        self.K = K
        self.save_dir = save_dir

        if isinstance(backbone_name, tf.keras.Model):
            self.backbone_model = backbone_name
        else:
            self.backbone_model = self._get_backbone(backbone_name, input_size, weights=backbone_weights)

        self.train_backbone = bool(train_backbone)
        if isinstance(self.backbone_model, tf.keras.Model):
            for layer in self.backbone_model.layers:
                layer.trainable = self.train_backbone

        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(0.5)  # regularisation on features

        # Concept heads: M sigmoid outputs
        self.attr_layers = [
            tf.keras.layers.Dense(
                1,
                activation="sigmoid",
                use_bias=True,
                name=f"attribute_output_concept_{c}",
            )
            for c in range(M)
        ]
        self.conc_attr_layer = tf.keras.layers.Concatenate(name="concatenated_attributes")

        if class_count is None:
            class_count = tf.ones((K,), dtype=tf.float32)
            
            
        # class_count: [K] number of training samples per class
        cc = tf.maximum(tf.cast(class_count, tf.float32), 1.0)
        # balanced weights: total/(K*count_k)
        w = tf.reduce_sum(cc) / (tf.cast(self.K, tf.float32) * cc)
        self.class_weights = w / tf.reduce_mean(w)  # normalize so mean weight = 1
        

        self.concept_class_priors = ConceptClassPriorsLayer(
            K,
            M,
            class_count,
            use_ema=use_ema_prior,
            momentum=ema_momentum,
            name="concept_class_priors",
        )
        self.class_layer = ClassificationLayer(K, M, name="class_layer")

        # Losses
        self.MSE = tf.keras.losses.MeanSquaredError()
        self.BCE = tf.keras.losses.BinaryCrossentropy()           # concept supervision
        self.CE = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.05)
        self.KL = tf.keras.losses.KLDivergence()
        
        
        # Decide concept mode
        if dataset_name.lower() in ["apy", "cub"]:
            self.concept_mode = "binary"
        elif dataset_name.lower() in ["awa", "awa2"]:
            self.concept_mode = "continuous"
            
        # Use one concept loss for optimization depending on concept type
        self.concept_loss_name = "bce" if self.concept_mode == "binary" else "mse"


        # Coefficients
        self.coeff_l_a_CE = float(coeff_l_a_CE)       # scales concept loss
        self.coeff_l_cls_CE = float(coeff_l_cls_CE)   # scales class CE
        self.coeff_l_prior_anch = float(coeff_prior_anch)  # scales prior anchor loss

        # Dynamic anchoring scalar
        self.anchor_alpha = self.add_weight(
            name="anchor_alpha",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False,
            dtype=tf.float32,
        )

        # Confusion matrices per epoch (for callbacks / logging)
        self.cm_train = self.add_weight(
            name="cm_train",
            shape=(K, K),
            initializer="zeros",
            trainable=False,
            dtype=tf.float32,
        )
        self.cm_val = self.add_weight(
            name="cm_val",
            shape=(K, K),
            initializer="zeros",
            trainable=False,
            dtype=tf.float32,
        )

        # For P(A_m|k) callbacks
        self._last_batch_attr_preds = None
        self._last_train_batch_labels = None
        
        
        
        self.train_cm_metrics = ConfusionMatrixMetrics(self.K, name="train_cm")
        self.val_cm_metrics   = ConfusionMatrixMetrics(self.K, name="val_cm")
        
        # proper epoch-level averages of losses too
        self.m_total_loss = tf.keras.metrics.Mean(name="total_loss")
        self.m_loss_c_CE = tf.keras.metrics.Mean(name="loss_c_CE")
        self.m_loss_a_MSE = tf.keras.metrics.Mean(name="loss_a_MSE")
        self.m_loss_a_BCE = tf.keras.metrics.Mean(name="loss_a_BCE")
        self.m_loss_prior = tf.keras.metrics.Mean(name="loss_prior_anch")
        self.m_anchor_alpha = tf.keras.metrics.Mean(name="anchor_alpha")


        # Keep regression-style trackers (they work for both)
        self.train_concept_reg = ConceptRegressionMetrics(self.M, name="train_concept_reg")
        self.val_concept_reg   = ConceptRegressionMetrics(self.M, name="val_concept_reg")
        
        # Only meaningful for binary concepts
        if self.concept_mode == "binary":
            self.train_concept_clf = ConceptClassificationMetrics(self.M, threshold=0.5, name="train_concept_clf")
            self.val_concept_clf   = ConceptClassificationMetrics(self.M, threshold=0.5, name="val_concept_clf")
        else:
            self.train_concept_clf = None
            self.val_concept_clf   = None
            

    @property
    def metrics(self):
        mets = [
            self.train_cm_metrics, self.val_cm_metrics,
            self.m_total_loss, self.m_loss_c_CE, self.m_loss_prior, self.m_anchor_alpha,
            self.train_concept_reg, self.val_concept_reg,
        ]
        if self.train_concept_clf is not None:
            mets += [self.train_concept_clf, self.val_concept_clf]
    
        #mets += [self.m_loss_a_MSE, self.m_loss_a_BCE]
        return mets




    def _get_backbone(self, name: str, input_shape, weights="imagenet"):
        name = name.lower()
        if name == "mobilenetv2":
            return tf.keras.applications.MobileNetV2(
                include_top=False,
                input_shape=input_shape,
                weights=weights,
            )
        if name == "resnet50":
            return tf.keras.applications.ResNet50(
                include_top=False,
                input_shape=input_shape,
                weights=weights,
            )
        if name == "efficientnetb0":
            return tf.keras.applications.EfficientNetB0(
                include_top=False,
                input_shape=input_shape,
                weights=weights,
            )
        if name == "inceptionv3":
            return tf.keras.applications.InceptionV3(
                include_top=False,
                input_shape=input_shape,
                weights=weights,
            )
        raise NotImplementedError(f"Backbone '{name}' not supported.")

    # ----------------------
    # Utilities
    # ----------------------
    def reset_epoch_confusions(self):
        self.cm_train.assign(tf.zeros_like(self.cm_train))
        self.cm_val.assign(tf.zeros_like(self.cm_val))

    @staticmethod
    def _confusion_metrics(y_true_idx, y_pred_idx, num_classes):
        """
        Compute balanced accuracy, macro-precision, macro-F1 from integer labels.
        """
        y_true_idx = tf.cast(y_true_idx, tf.int32)
        y_pred_idx = tf.cast(y_pred_idx, tf.int32)

        cm = tf.math.confusion_matrix(
            y_true_idx,
            y_pred_idx,
            num_classes=num_classes,
            dtype=tf.float32,
        )  # [K,K]
        eps = 1e-8

        tp = tf.linalg.diag_part(cm)          # [K]
        row_sum = tf.reduce_sum(cm, axis=1)   # [K]  true counts
        col_sum = tf.reduce_sum(cm, axis=0)   # [K]  pred counts

        recall_c = tf.where(row_sum > 0.0, tp / row_sum, tf.zeros_like(tp))
        prec_c = tf.where(col_sum > 0.0, tp / col_sum, tf.zeros_like(tp))

        mask_true = row_sum > 0.0
        mask_pred = col_sum > 0.0
        mask_f1 = mask_true & ((prec_c + recall_c) > 0.0)

        bal_acc = tf.cond(
            tf.reduce_any(mask_true),
            lambda: tf.reduce_mean(tf.boolean_mask(recall_c, mask_true)),
            lambda: tf.constant(0.0, tf.float32),
        )
        macro_prec = tf.cond(
            tf.reduce_any(mask_pred),
            lambda: tf.reduce_mean(tf.boolean_mask(prec_c, mask_pred)),
            lambda: tf.constant(0.0, tf.float32),
        )

        f1_c = tf.where(
            (prec_c + recall_c) > 0.0,
            2.0 * prec_c * recall_c / (prec_c + recall_c + eps),
            tf.zeros_like(tp),
        )
        macro_f1 = tf.cond(
            tf.reduce_any(mask_f1),
            lambda: tf.reduce_mean(tf.boolean_mask(f1_c, mask_f1)),
            lambda: tf.constant(0.0, tf.float32),
        )

        return bal_acc, macro_prec, macro_f1
    
    
    def _compute_concept_losses(self, concept_annot, concepts_pred):
        concept_annot = tf.cast(concept_annot, tf.float32)
        concepts_bce = self.BCE(concept_annot, concepts_pred)
        concepts_mse = self.MSE(concept_annot, concepts_pred)
    
        # Choose which one drives optimization
        concept_opt_loss = concepts_bce if self.concept_loss_name == "bce" else concepts_mse
        return concept_opt_loss, concepts_bce, concepts_mse


    # ----------------------
    # Forward
    # ----------------------
    def call(self, inputs, training=False):
        """
        inputs: (input_data, labels, attr_ann)
          - input_data: [B,H,W,3]
          - labels:     [B]       (int)
          - attr_ann:   [B,M]
        """
        input_data, labels, attr_ann = inputs

        # Backbone
        x = self.backbone_model(input_data, training=training)
        if len(x.shape) == 4:
            x = self.global_avg_pool(x)
        x = self.dropout(x, training=training)

        # Concept predictions
        per_concepts_pred = [head(x) for head in self.attr_layers]   # list of [B,1]
        concepts_pred = self.conc_attr_layer(per_concepts_pred)      # [B,M]

        # Priors P(A_m|y=k)
        P_pred_used, P_true_batch = self.concept_class_priors(
            [concepts_pred, labels, attr_ann],
            training=training,
        )  # each [M,K]

        # Transparent classifier over concepts
        class_logits, class_probs = self.class_layer(concepts_pred, training=training)

        return (
            per_concepts_pred,
            concepts_pred,
            (P_pred_used, P_true_batch),
            class_logits,
            class_probs,
        )

    # ----------------------
    # Custom train / test
    # ----------------------
    def train_step(self, data):
        """
        data from tf.data: (images, labels, concepts)
          - images:   [B,H,W,3]
          - labels:   [B]
          - concepts: [B,M]
        """
        input_data, class_labels, concept_annot = data
        class_labels_int = tf.cast(class_labels, tf.int32)
        y_oh = tf.one_hot(class_labels_int, depth=self.K, dtype=tf.float32)

        with tf.GradientTape() as tape:
            _, concepts_pred, (P_pred, P_true), _, class_probs = self(
                [input_data, class_labels, concept_annot],
                training=True,
            )

            #frac_pos = tf.reduce_mean(tf.cast(concepts_pred > 0.5, tf.float32))
            #tf.print("============= frac(pred>0.5)=", frac_pos)
    
            # For P(A_m|k) callbacks
            self._last_batch_attr_preds = concepts_pred
            self._last_train_batch_labels = class_labels_int

            # Concept losses
            concept_opt_loss, concepts_bce, concepts_mse = self._compute_concept_losses(concept_annot, concepts_pred)



            # Weighted class CE 
            per_sample_ce = tf.keras.losses.categorical_crossentropy(y_oh, class_probs, from_logits=False, label_smoothing=0.05)
            sample_w = tf.gather(self.class_weights, class_labels_int)  # [B]
            classes_loss = tf.reduce_mean(per_sample_ce * sample_w)               

            # Dynamic anchoring
            alpha = tf.clip_by_value(self.anchor_alpha, 0.0, 1.0)  # scalar

            # Mix TRUE and PRED priors as target
            target_prior = (1.0 - alpha) * tf.stop_gradient(P_true) + alpha * tf.stop_gradient(P_pred)

            # Avoid numerical issues
            target_prior = tf.clip_by_value(target_prior, 1e-4, 1.0 - 1e-4)

            # logit(P) = log(P) - log(1-P)
            target_logits = tf.math.log(target_prior) - tf.math.log(1.0 - target_prior)  # [M,K]

            gamma_logits = self.class_layer.gamma  # [M,K]

            # Anchor loss: MSE(gamma, logit(P(A_m|y=k)))
            prior_anch_loss = tf.reduce_mean(tf.square(gamma_logits - target_logits))

            # Regularization losses (e.g. L2 on gamma)
            reg_loss = 0#tf.add_n(self.losses) if self.losses else 0.0

            total_loss = (
                self.coeff_l_a_CE * concept_opt_loss
                + self.coeff_l_cls_CE * classes_loss
                + self.coeff_l_prior_anch * prior_anch_loss
            )

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        y_hat = tf.argmax(class_probs, axis=1, output_type=tf.int32)

        # update epoch confusion-matrix metric
        self.train_cm_metrics.update_state(class_labels_int, y_hat)
        
        # update epoch-mean losses
        self.m_total_loss.update_state(total_loss)
        self.m_loss_a_BCE.update_state(concepts_bce)
        self.m_loss_a_MSE.update_state(concepts_mse)
        self.m_loss_c_CE.update_state(classes_loss)
        self.m_loss_prior.update_state(prior_anch_loss)
        self.m_anchor_alpha.update_state(alpha)
        
        cm_res = self.train_cm_metrics.result()
        
        
        self.train_concept_reg.update_state(concept_annot, concepts_pred)
        if self.train_concept_clf is not None:
            self.train_concept_clf.update_state(concept_annot, concepts_pred)
        
        reg = self.train_concept_reg.result()


                
        out = {
            "acc": cm_res["accuracy"],
            "bal_acc": cm_res["balanced_accuracy"],
            "mac_pr": cm_res["macro_precision"],
            "mac_rec": cm_res["macro_recall"],
            "mac_f1": cm_res["macro_f1"],
            "concept_mse_micro": reg["mse_micro"],
            "concept_mse_macro": reg["mse_macro"],
            "concept_bce_micro": reg["bce_micro"],
            "concept_bce_macro": reg["bce_macro"],
            "loss_a_BCE": self.m_loss_a_BCE.result(),
            "loss_a_MSE": self.m_loss_a_MSE.result(),
            "loss_c_CE": self.m_loss_c_CE.result(),
            "loss_prior_anch": self.m_loss_prior.result(),
            "total_loss": self.m_total_loss.result(),
            "anchor_alpha": self.m_anchor_alpha.result(),
        }
        
        if self.train_concept_clf is not None:
            clf = self.train_concept_clf.result()
            out["concept_f1_macro"] = clf["c_f1_macro"]
            
        return out


        """# Metrics
        y_hat = tf.argmax(class_probs, axis=1, output_type=tf.int32)
        acc_cls = tf.reduce_mean(tf.cast(tf.equal(y_hat, class_labels_int), tf.float32))

        bal_acc_cls, mac_pr_cls, mac_f1_cls = self._confusion_metrics(
            class_labels_int, y_hat, self.K)

        cm_batch = tf.math.confusion_matrix(
            class_labels_int,
            y_hat,
            num_classes=self.K,
            dtype=tf.float32,
        )
        self.cm_train.assign_add(cm_batch)

        return {
            "acc": acc_cls,
            #"bal_acc": bal_acc_cls,
            #"mac_pr": mac_pr_cls,
            #"mac_f1": mac_f1_cls,
            "loss_a_BCE": concepts_bce,
            "loss_a_MSE": concepts_mse,
            "loss_c_CE": classes_loss,
            "loss_prior_anch": prior_anch_loss,
            "total_loss": total_loss,
            "anchor_alpha": alpha,
        }"""

    def test_step(self, data):
        input_data, class_labels, concept_annot = data

        _, concepts_pred, _, _, class_probs = self([input_data, class_labels, concept_annot], training=False,)
        
        y_true = tf.cast(class_labels, tf.int32)
        y_hat = tf.argmax(class_probs, axis=1, output_type=tf.int32)
        
        # update epoch confusion-matrix metric for validation
        self.val_cm_metrics.update_state(y_true, y_hat)
    
        cm_res = self.val_cm_metrics.result()
        
        self.val_concept_reg.update_state(concept_annot, concepts_pred)
        if self.val_concept_clf is not None:
            self.val_concept_clf.update_state(concept_annot, concepts_pred)
        
        reg = self.val_concept_reg.result()
        cm_res = self.val_cm_metrics.result()
        
        out = {
            "acc": cm_res["accuracy"],
            "bal_acc": cm_res["balanced_accuracy"],
            "mac_pr": cm_res["macro_precision"],
            "mac_rec": cm_res["macro_recall"],
            "mac_f1": cm_res["macro_f1"],
            "concept_mse_micro": reg["mse_micro"],
            "concept_mse_macro": reg["mse_macro"],
            "concept_bce_micro": reg["bce_micro"],
            "concept_bce_macro": reg["bce_macro"],
        }
        if self.val_concept_clf is not None:
            clf = self.val_concept_clf.result()
            out["concept_f1_macro"] = clf["c_f1_macro"]
        
        return out 
        

    
    #averaging per-batch balanced accuracies (which what the KERAS logs does) is not mathematically equivalent 
    #to computing balanced accuracy on the full epoch (which is what our ConfMatSaver callback does)




class ConceptRegressionMetrics(tf.keras.metrics.Metric):
    """
    Tracks per-concept MSE and BCE over an epoch.
    Produces:
      - mse_micro: overall MSE over all (samples, concepts)  
      - mse_macro: mean over concepts of per-concept MSE
      - bce_micro: overall BCE
      - bce_macro: mean over concepts of per-concept BCE
    """
    def __init__(self, num_concepts: int, name="concept_reg", **kwargs):
        super().__init__(name=name, **kwargs)
        self.M = int(num_concepts)
        # SSE and counts for MSE
        self.sse = self.add_weight(name="sse", shape=(self.M,), initializer="zeros", dtype=tf.float32)
        self.n   = self.add_weight(name="n",   shape=(self.M,), initializer="zeros", dtype=tf.float32)
        # Sum BCE and counts for BCE
        self.bce_sum = self.add_weight(name="bce_sum", shape=(self.M,), initializer="zeros", dtype=tf.float32)
        self.bce_n   = self.add_weight(name="bce_n",   shape=(self.M,), initializer="zeros", dtype=tf.float32)

    def reset_state(self):
        self.sse.assign(tf.zeros_like(self.sse))
        self.n.assign(tf.zeros_like(self.n))
        self.bce_sum.assign(tf.zeros_like(self.bce_sum))
        self.bce_n.assign(tf.zeros_like(self.bce_n))

    def update_state(self, a_true, a_pred):
        # a_true, a_pred: [B, M]
        a_true = tf.cast(a_true, tf.float32)
        a_pred = tf.cast(a_pred, tf.float32)

        # MSE accumulation
        se = tf.square(a_true - a_pred)            # [B,M]
        self.sse.assign_add(tf.reduce_sum(se, axis=0))  # sum over batch => [M]
        self.n.assign_add(tf.cast(tf.shape(a_true)[0], tf.float32) * tf.ones((self.M,), tf.float32))

        # BCE accumulation (per-concept)
        eps = 1e-7
        p = tf.clip_by_value(a_pred, eps, 1.0 - eps)
        bce = -(a_true * tf.math.log(p) + (1.0 - a_true) * tf.math.log(1.0 - p))  # [B,M]
        self.bce_sum.assign_add(tf.reduce_sum(bce, axis=0))
        self.bce_n.assign_add(tf.cast(tf.shape(a_true)[0], tf.float32) * tf.ones((self.M,), tf.float32))

    def result(self):
        eps = 1e-8
        mse_per_concept = self.sse / (self.n + eps)         # [M]
        bce_per_concept = self.bce_sum / (self.bce_n + eps) # [M]

        mse_micro = tf.reduce_sum(self.sse) / (tf.reduce_sum(self.n) + eps)
        mse_macro = tf.reduce_mean(mse_per_concept)

        bce_micro = tf.reduce_sum(self.bce_sum) / (tf.reduce_sum(self.bce_n) + eps)
        bce_macro = tf.reduce_mean(bce_per_concept)

        return {
            "mse_micro": mse_micro,
            "mse_macro": mse_macro,
            "bce_micro": bce_micro,
            "bce_macro": bce_macro,
        }

    def per_concept(self):
        """Optional: access per-concept arrays for dashboards/callbacks."""
        eps = 1e-8
        return {
            "mse": self.sse / (self.n + eps),
            "bce": self.bce_sum / (self.bce_n + eps),
        }



class ConceptClassificationMetrics(tf.keras.metrics.Metric):
    """
    For binary concepts, threshold predictions and compute per-concept:
      precision, recall, f1
    and macro averages across concepts.

    Also tracks prevalence (support positives).
    """
    def __init__(self, num_concepts: int, threshold=0.5, name="concept_clf", **kwargs):
        super().__init__(name=name, **kwargs)
        self.M = int(num_concepts)
        self.t = float(threshold)

        self.tp = self.add_weight(name="tp", shape=(self.M,), initializer="zeros", dtype=tf.float32)
        self.fp = self.add_weight(name="fp", shape=(self.M,), initializer="zeros", dtype=tf.float32)
        self.fn = self.add_weight(name="fn", shape=(self.M,), initializer="zeros", dtype=tf.float32)
        self.pos = self.add_weight(name="pos", shape=(self.M,), initializer="zeros", dtype=tf.float32)  # positives in GT
        self.total = self.add_weight(name="total", shape=(self.M,), initializer="zeros", dtype=tf.float32)

    def reset_state(self):
        for w in (self.tp, self.fp, self.fn, self.pos, self.total):
            w.assign(tf.zeros_like(w))

    def update_state(self, a_true, a_pred):
        a_true = tf.cast(a_true, tf.float32)
        a_hat = tf.cast(a_pred >= self.t, tf.float32)

        tp = a_hat * a_true
        fp = a_hat * (1.0 - a_true)
        fn = (1.0 - a_hat) * a_true

        self.tp.assign_add(tf.reduce_sum(tp, axis=0))
        self.fp.assign_add(tf.reduce_sum(fp, axis=0))
        self.fn.assign_add(tf.reduce_sum(fn, axis=0))

        self.pos.assign_add(tf.reduce_sum(a_true, axis=0))
        self.total.assign_add(tf.cast(tf.shape(a_true)[0], tf.float32) * tf.ones((self.M,), tf.float32))

    def result(self):
        eps = 1e-8
        prec = self.tp / (self.tp + self.fp + eps)
        rec  = self.tp / (self.tp + self.fn + eps)
        f1   = 2.0 * prec * rec / (prec + rec + eps)

        macro_prec = tf.reduce_mean(prec)
        macro_rec  = tf.reduce_mean(rec)
        macro_f1   = tf.reduce_mean(f1)

        prevalence = self.pos / (self.total + eps)  # fraction of samples where concept is positive

        return {
            "c_prec_macro": macro_prec,
            "c_rec_macro": macro_rec,
            "c_f1_macro": macro_f1,
            "c_prev_mean": tf.reduce_mean(prevalence),
        }

    def per_concept(self):
        eps = 1e-8
        prec = self.tp / (self.tp + self.fp + eps)
        rec  = self.tp / (self.tp + self.fn + eps)
        f1   = 2.0 * prec * rec / (prec + rec + eps)
        prevalence = self.pos / (self.total + eps)
        return {"precision": prec, "recall": rec, "f1": f1, "prevalence": prevalence}
