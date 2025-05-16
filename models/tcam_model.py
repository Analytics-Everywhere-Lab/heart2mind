from keras_tuner import HyperModel
from keras.layers import Input, Conv1D, BatchNormalization, ReLU, Add, MaxPooling1D, Dense, LSTM, Dropout, \
    Bidirectional, LayerNormalization, MultiHeadAttention, Flatten
from keras.models import Model
from keras.regularizers import l2
from keras_tuner import HyperModel, HyperParameters

# Define sequence length
sequence_length = 50
def initialize_base_model(hp):
    tcam_model = TCAM.from_hyperparams(hp)
    return tcam_model

class TCAM(HyperModel):
    def __init__(self):
        super(TCAM, self).__init__()

    def build(self, hp):
        inputs = Input(shape=(sequence_length, 1))

        # Temporal Convolutional Network (TCN)
        filters_1 = hp.Int('filters_1', min_value=32, max_value=512, step=32)
        filters_2 = hp.Int('filters_2', min_value=64, max_value=1024, step=64)
        filters_3 = hp.Int('filters_3', min_value=64, max_value=1024, step=64)
        filters_4 = hp.Int('filters_4', min_value=64, max_value=1024, step=64)
        kernel_size = hp.Int('kernel_size', min_value=2, max_value=10, step=1)
        lstm_units = hp.Int('lstm_units', min_value=64, max_value=1024, step=64)
        attention_heads = hp.Int('attention_heads', min_value=2, max_value=32, step=2)
        attention_key_dim = hp.Int('attention_key_dim', min_value=32, max_value=512, step=32)
        dense_units = hp.Int('dense_units', min_value=64, max_value=512, step=64)
        dropout_rate = hp.Float('dropout', 0.2, 0.5, step=0.1)

        x = Conv1D(filters_1,
                   kernel_size=kernel_size,
                   strides=1,
                   padding='same',
                   activation='relu',
                   kernel_regularizer=l2(1e-4))(inputs)

        x = BatchNormalization()(x)
        x = Conv1D(filters_2, kernel_size=kernel_size, strides=1, padding='same', activation='relu',
                   kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

        # First Residual Block
        shortcut = Conv1D(filters_2, kernel_size=1, padding='same')(inputs)
        shortcut = MaxPooling1D(pool_size=2)(shortcut)

        x = Add()([shortcut, x])
        x = ReLU()(x)

        # Additional Residual Block
        x_shortcut = x
        x = Conv1D(filters_3,
                   kernel_size=kernel_size,
                   strides=1,
                   padding='same',
                   activation='relu',
                   kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters_4, kernel_size=kernel_size, strides=1, padding='same', activation='relu',
                   kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Add()([x_shortcut, x])
        x = ReLU()(x)

        # Bidirectional LSTM with Attention
        x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
        attention = MultiHeadAttention(num_heads=attention_heads, key_dim=attention_key_dim)(x, x)
        x = Add()([x, attention])
        x = LayerNormalization()(x)

        # Flatten and Dense layers
        x = Flatten()(x)
        x = Dense(dense_units, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(2, activation='softmax')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod
    def from_hyperparams(hyperparams):
        hp = HyperParameters()
        for key, value in hyperparams.items():
            hp.Fixed(key, value)
        return TCAM().build(hp)
