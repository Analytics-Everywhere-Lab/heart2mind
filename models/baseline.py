from keras.models import Model
from keras.layers import (
    Input,
    LSTM,
    GRU,
    Conv1D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
)
from keras.optimizers import Adam
from keras_tuner import HyperModel, HyperParameters
from keras.layers import MultiHeadAttention, LayerNormalization, Add


def initialize_GRU_model(hp):
    gru_model = GRUHyperModel.from_hyperparams(hp)
    return gru_model


def initialize_LSTM_model(hp):
    lstm_model = LSTMHyperModel.from_hyperparams(hp)
    return lstm_model


def initialize_CNN_model(hp):
    cnn_model = CNNHyperModel.from_hyperparams(hp)
    return cnn_model


def initialize_Transformer_model(hp):
    transformer_model = TransformerHyperModel.from_hyperparams(hp)
    return transformer_model


class LSTMHyperModel(HyperModel):
    def __init__(self):
        super(LSTMHyperModel, self).__init__()

    def build(self, hp):
        inputs = Input(shape=(50, 1))
        x = LSTM(hp.Int("units_1", 64, 1024, step=64), return_sequences=True)(inputs)
        x = LSTM(hp.Int("units_2", 64, 1024, step=64))(x)
        x = Dense(hp.Int("dense_units", 64, 1024, step=64), activation="relu")(x)
        x = Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1))(x)
        outputs = Dense(2, activation="softmax")(x)
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    @staticmethod
    def from_hyperparams(hyperparams):
        hp = HyperParameters()
        for key, value in hyperparams.items():
            hp.Fixed(key, value)
        return LSTMHyperModel().build(hp)


class GRUHyperModel(HyperModel):
    def __init__(self):
        super(GRUHyperModel, self).__init__()

    def build(self, hp):
        inputs = Input(shape=(50, 1))
        x = GRU(hp.Int("units_1", 64, 1024, step=64), return_sequences=True)(inputs)
        x = GRU(hp.Int("units_2", 64, 1024, step=64))(x)
        x = Dense(hp.Int("dense_units", 64, 1024, step=64), activation="relu")(x)
        x = Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1))(x)
        outputs = Dense(2, activation="softmax")(x)
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    @staticmethod
    def from_hyperparams(hyperparams):
        hp = HyperParameters()
        for key, value in hyperparams.items():
            hp.Fixed(key, value)
        return GRUHyperModel().build(hp)


class CNNHyperModel(HyperModel):
    def __init__(self):
        super(CNNHyperModel, self).__init__()

    def build(self, hp):
        inputs = Input(shape=(50, 1))
        x = Conv1D(
            hp.Int("filters_1", 64, 1024, step=64), kernel_size=3, activation="relu"
        )(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(
            hp.Int("filters_2", 64, 1024, step=64), kernel_size=3, activation="relu"
        )(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(hp.Int("dense_units", 64, 1024, step=64), activation="relu")(x)
        x = Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1))(x)
        outputs = Dense(2, activation="softmax")(x)
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    @staticmethod
    def from_hyperparams(hyperparams):
        hp = HyperParameters()
        for key, value in hyperparams.items():
            hp.Fixed(key, value)
        return CNNHyperModel().build(hp)


class TransformerHyperModel(HyperModel):
    def __init__(self):
        super(TransformerHyperModel, self).__init__()

    def build(self, hp):
        inputs = Input(shape=(50, 1))
        attn_output = MultiHeadAttention(
            num_heads=4, key_dim=hp.Int("key_dim", 64, 1024, step=64)
        )(inputs, inputs)
        attn_output = Dropout(0.1)(attn_output)
        out1 = Add()([inputs, attn_output])
        out1 = LayerNormalization(epsilon=1e-6)(out1)

        ffn_output = Dense(hp.Int("ffn_units_1", 64, 1024, step=64), activation="relu")(
            out1
        )
        ffn_output = Dense(hp.Int("ffn_units_2", 64, 1024, step=64))(ffn_output)
        ffn_output = Dropout(0.1)(ffn_output)
        out2 = Add()([out1, ffn_output])
        out2 = LayerNormalization(epsilon=1e-6)(out2)

        x = Flatten()(out2)
        x = Dense(hp.Int("dense_units", 64, 1024, step=64), activation="relu")(x)
        x = Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1))(x)
        outputs = Dense(2, activation="softmax")(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    @staticmethod
    def from_hyperparams(hyperparams):
        hp = HyperParameters()
        for key, value in hyperparams.items():
            hp.Fixed(key, value)
        return TransformerHyperModel().build(hp)
