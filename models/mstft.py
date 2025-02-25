import math
import tensorflow as tf
from keras import layers, Model
from tensorflow_addons.metrics import F1Score
from keras.metrics import Precision, Recall, AUC, BinaryAccuracy
from tensorflow_addons.layers import SpectralNormalization
from keras.regularizers import l2
import tensorflow_addons as tfa


class MSTFT:
    def __init__(
        self,
        sequence_length,
        l2_weight=1e-4,
        dropout_rate=0.1,
        num_heads=8,
        key_dim=64,
        project_dim=512,
        temporal_filters=256,
        dilation_rate=3,
        wavelet_filters=256,
        learning_rate=1e-5,
        weight_decay=1e-5,
        num_temporal_blocks=2,
        num_freq_blocks=2,
        expansion_factor=2,
        survival_prob=0.8,
        attention_dropout=0.1,
        ffn_dropout=0.2,
        use_aux_loss=True,
        mixup_alpha=0.3,
    ):
        self.sequence_length = sequence_length
        self.l2_weight = l2_weight
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.project_dim = project_dim
        self.temporal_filters = temporal_filters
        self.dilation_rate = dilation_rate
        self.wavelet_filters = wavelet_filters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_temporal_blocks = num_temporal_blocks
        self.num_freq_blocks = num_freq_blocks
        self.expansion_factor = expansion_factor
        self.survival_prob = survival_prob
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.use_aux_loss = use_aux_loss
        self.mixup_alpha = mixup_alpha

    def stochastic_skip(self, x, res):
        """Implements stochastic depth for residual connections"""
        if x.shape[-1] != res.shape[-1]:
            res = layers.Dense(x.shape[-1])(res)
        if tf.keras.backend.learning_phase():
            survival = tf.random.uniform([], 0, 1) < self.survival_prob
            return tf.cond(survival, lambda: x + res, lambda: res)
        return x + res

    def add_positional_encoding(self, x):
        """Positional encoding with feature projection"""
        # First project to higher dimension
        x = layers.Conv1D(64, 1)(x)
        feature_dim = x.shape[-1]
        position = tf.range(
            start=0, limit=self.sequence_length, delta=1, dtype=tf.float32
        )
        div_term = tf.exp(
            tf.range(0, feature_dim, 2, dtype=tf.float32)
            * -(math.log(10000.0) / feature_dim)
        )
        sin_part = tf.sin(tf.expand_dims(position, 1) * div_term)
        cos_part = tf.cos(tf.expand_dims(position, 1) * div_term)
        pos_encoding = tf.stack([sin_part, cos_part], axis=-1)
        pos_encoding = tf.reshape(pos_encoding, [1, self.sequence_length, feature_dim])
        return x + pos_encoding

    def build(self):
        inputs = layers.Input(shape=(self.sequence_length, 1))
        x = layers.GaussianNoise(stddev=0.2)(inputs)
        x = self.add_positional_encoding(x)
        # --- Branch 1: Multi-Scale Temporal Convolutions ---
        x_temporal = x
        for i in range(self.num_temporal_blocks):
            res = x_temporal
            x_temporal = SpectralNormalization(
                layers.Conv1D(
                    filters=self.temporal_filters // (2**i),
                    kernel_size=3,
                    dilation_rate=2**i,
                    padding="causal",
                    kernel_regularizer=l2(self.l2_weight),
                )
            )(x_temporal)

            x_temporal = tfa.layers.GroupNormalization(8)(x_temporal)
            x_temporal = layers.Activation("gelu")(x_temporal)

            # Residual connection
            if res.shape[-1] != x_temporal.shape[-1]:
                res = SpectralNormalization(layers.Conv1D(x_temporal.shape[-1], 1))(res)

            x_temporal = self.stochastic_skip(x_temporal, res)
            res = layers.AveragePooling1D(2)(x_temporal)
            x_temporal = layers.SpatialDropout1D(0.2)(x_temporal)

        # --- Branch 2: Learnable Wavelet Transform (Frequency Domain) ---
        x_freq = layers.Conv1D(64, 11, padding="same")(x)
        for j in range(self.num_freq_blocks):
            x_freq = layers.SeparableConv1D(
                filters=self.wavelet_filters * (j + 1),
                kernel_size=5,
                depth_multiplier=2,
                padding="same",
                activation="gelu",
                depthwise_regularizer=tf.keras.regularizers.l2(self.l2_weight),
                pointwise_regularizer=tf.keras.regularizers.l2(self.l2_weight),
            )(x_freq)

            x_freq = layers.BatchNormalization()(x_freq)

            # Adaptive frequency pooling
            if j % 2 == 0:
                x_freq = tfa.layers.AdaptiveAveragePooling1D(x_temporal.shape[1])(x_freq)

        # --- Cross-Attention Fusion ---
        x_temporal = layers.Dense(self.project_dim)(x_temporal)
        x_freq = layers.Dense(self.project_dim)(x_freq)

        q = layers.Dense(self.key_dim)(x_temporal)
        k = layers.Dense(self.key_dim)(x_freq)
        v = layers.Dense(self.key_dim)(x_freq)

        fused = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim // self.num_heads,
            value_dim=self.key_dim // self.num_heads,
        )(q, k, v)

        fused = layers.Dense(self.project_dim)(fused)
        fused = layers.Concatenate()([fused, x_temporal, x_freq])
        fused = layers.LayerNormalization()(fused)

        # Transformer Encoder Block
        for _ in range(2):
            attn = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.key_dim // self.num_heads,
                value_dim=self.key_dim // self.num_heads,
            )(fused, fused)
            attn = layers.Dense(self.project_dim)(attn)
            fused = self.stochastic_skip(attn, fused)

            # Gated FFN
            ffn = layers.Dense(fused.shape[-1] * 2, activation='gelu')(fused)
            ffn = layers.Dense(fused.shape[-1])(ffn)
    
            fused = self.stochastic_skip(ffn, fused)
            fused = layers.LayerNormalization()(fused)

        # --- Classifier Head ---
        x = layers.Concatenate()(
            [layers.GlobalAveragePooling1D()(fused), layers.GlobalMaxPooling1D()(fused)]
        )
        x = layers.BatchNormalization()(x)

        for _ in range(2):
            residual = x
            x = layers.Dense(self.project_dim // 2, activation="gelu")(x)
            x = layers.Dropout(0.3)(x)
            residual = layers.Dense(self.project_dim // 2)(residual)
            x = layers.Add()([x, residual])
            x = layers.BatchNormalization()(x)

        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                clipnorm=1.0,
            ),
            loss=tf.keras.losses.BinaryFocalCrossentropy(),
            metrics=[
                BinaryAccuracy(name="accuracy"),
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(name="auc"),
                F1Score(
                    num_classes=1, average="weighted", threshold=0.5, name="f1_score"
                ),
            ],
        )
        return model
