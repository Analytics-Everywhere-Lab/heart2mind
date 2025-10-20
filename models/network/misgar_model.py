from keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    Flatten,
    Lambda,
    Concatenate,
    PReLU,
)
from tensorflow_addons.layers import InstanceNormalization
from keras.models import Model
from tensorflow import add


def initialize_misgar_model():
    model_builder = MisgarModel()
    return model_builder.build()


class MisgarModel:
    def __init__(self):
        self.input_shape = (1440, 1)

    def build(self):
        inputs = Input(shape=self.input_shape, name="input_3")

        x = Lambda(lambda x: x[:, ::1, :])(inputs)
        y = Lambda(lambda x: x[:, ::1, :])(x)

        x = Conv1D(512, 13, padding="same", name="conv1d_13")(inputs)
        x = Conv1D(512, 15, padding="same", name="conv1d_15")(x)
        x = MaxPooling1D(pool_size=2, name="max_pooling1d_12")(x)
        x = MaxPooling1D(pool_size=2, name="max_pooling1d_14")(x)

        y = Conv1D(256, 14, padding="same", name="conv1d_14")(y)
        y = Conv1D(256, 16, padding="same", name="conv1d_16")(y)
        y = MaxPooling1D(pool_size=2, name="max_pooling1d_13")(y)
        y = MaxPooling1D(pool_size=2, name="max_pooling1d_15")(y)

        z = Concatenate(name="concatenate_2")([x, y])

        z = Conv1D(128, 17, padding="same", name="conv1d_17")(z)
        z = InstanceNormalization(name="instance_normalization_7")(z)
        z = PReLU(name="p_re_lu_5")(z)
        z = Dropout(0.5, name="dropout_5")(z)
        z = MaxPooling1D(pool_size=2, name="max_pooling1d_16")(z)

        z = MultiHeadAttention(num_heads=4, key_dim=128, name="multi_head_attention_1")(
            z, z
        )
        z = add(z, z, name="tf__operators__add_1")
        z = LayerNormalization(name="layer_normalization_1")(z)

        z = Conv1D(512, 18, padding="same", name="conv1d_18")(z)
        z = InstanceNormalization(name="instance_normalization_8")(z)
        z = PReLU(name="p_re_lu_6")(z)
        z = Dropout(0.6, name="dropout_6")(z)
        z = MaxPooling1D(pool_size=2, name="max_pooling1d_17")(z)

        z = Dense(32, activation="relu", name="dense_10")(z)
        z = Dense(16, activation="relu", name="dense_11")(z)
        z = Dense(8, activation="relu", name="dense_12")(z)
        z = Dense(4, activation="relu", name="dense_13")(z)
        z = InstanceNormalization(name="instance_normalization_9")(z)

        z = Flatten(name="flatten_2")(z)
        outputs = Dense(2, activation="softmax", name="dense_14")(z)

        model = Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model
