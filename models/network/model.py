from keras_tuner import HyperModel
from keras.layers import Input, Conv1D, BatchNormalization, ReLU, Add, MaxPooling1D, Dense, LSTM, Dropout, \
    Bidirectional, LayerNormalization, MultiHeadAttention, Flatten, Lambda, Concatenate, PReLU
from tensorflow_addons.layers import InstanceNormalization
from keras.models import Model
from keras.regularizers import l2
from tensorflow import add
# Define sequence length
sequence_length = 50


class AdvancedHyperModel(HyperModel):
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

        x = Conv1D(filters_1, kernel_size=kernel_size, strides=1, padding='same', activation='relu',
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
        x = Conv1D(filters_3, kernel_size=kernel_size, strides=1, padding='same', activation='relu',
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


class AdvancedHyperModelDict:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def build(self):
        inputs = Input(shape=(sequence_length, 1))

        # Temporal Convolutional Network (TCN)
        filters_1 = self.hyperparameters['filters_1']
        filters_2 = self.hyperparameters['filters_2']
        filters_3 = self.hyperparameters['filters_3']
        filters_4 = self.hyperparameters['filters_4']
        kernel_size = self.hyperparameters['kernel_size']

        x = Conv1D(filters_1, kernel_size=kernel_size, strides=1, padding='same', activation='relu',
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
        x = Conv1D(filters_3, kernel_size=kernel_size, strides=1, padding='same', activation='relu',
                   kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters_4, kernel_size=kernel_size, strides=1, padding='same', activation='relu',
                   kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Add()([x_shortcut, x])
        x = ReLU()(x)

        # Bidirectional LSTM with Attention
        lstm_units = self.hyperparameters['lstm_units']
        attention_heads = self.hyperparameters['attention_heads']
        attention_key_dim = self.hyperparameters['attention_key_dim']

        x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
        attention = MultiHeadAttention(num_heads=attention_heads, key_dim=attention_key_dim)(x, x)
        x = Add()([x, attention])
        x = LayerNormalization()(x)

        # Flatten and Dense layers
        dense_units = self.hyperparameters['dense_units']
        dropout_rate = self.hyperparameters['dropout']

        x = Flatten()(x)
        x = Dense(dense_units, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(2, activation='softmax')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model


class CustomModelArchitecture:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self):
        inputs = Input(shape=self.input_shape, name='input_3')

        x = Lambda(lambda x: x[:, ::1, :])(inputs)
        y = Lambda(lambda x: x[:, ::1, :])(x)

        x = Conv1D(512, 13, padding='same', name='conv1d_13')(inputs)
        x = Conv1D(512, 15, padding='same', name='conv1d_15')(x)
        x = MaxPooling1D(pool_size=2, name='max_pooling1d_12')(x)
        x = MaxPooling1D(pool_size=2, name='max_pooling1d_14')(x)

        y = Conv1D(256, 14, padding='same', name='conv1d_14')(y)
        y = Conv1D(256, 16, padding='same', name='conv1d_16')(y)
        y = MaxPooling1D(pool_size=2, name='max_pooling1d_13')(y)
        y = MaxPooling1D(pool_size=2, name='max_pooling1d_15')(y)

        z = Concatenate(name='concatenate_2')([x, y])

        z = Conv1D(128, 17, padding='same', name='conv1d_17')(z)
        z = InstanceNormalization(name='instance_normalization_7')(z)
        z = PReLU(name='p_re_lu_5')(z)
        z = Dropout(0.5, name='dropout_5')(z)
        z = MaxPooling1D(pool_size=2, name='max_pooling1d_16')(z)

        z = MultiHeadAttention(num_heads=4, key_dim=128, name='multi_head_attention_1')(z, z)
        z = add(z, z, name='tf__operators__add_1')
        z = LayerNormalization(name='layer_normalization_1')(z)

        z = Conv1D(512, 18, padding='same', name='conv1d_18')(z)
        z = InstanceNormalization(name='instance_normalization_8')(z)
        z = PReLU(name='p_re_lu_6')(z)
        z = Dropout(0.6, name='dropout_6')(z)
        z = MaxPooling1D(pool_size=2, name='max_pooling1d_17')(z)

        z = Dense(32, activation='relu', name='dense_10')(z)
        z = Dense(16, activation='relu', name='dense_11')(z)
        z = Dense(8, activation='relu', name='dense_12')(z)
        z = Dense(4, activation='relu', name='dense_13')(z)
        z = InstanceNormalization(name='instance_normalization_9')(z)

        z = Flatten(name='flatten_2')(z)
        outputs = Dense(2, activation='softmax', name='dense_14')(z)

        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model