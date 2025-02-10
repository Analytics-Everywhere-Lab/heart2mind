import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from misgar_model import CustomModelArchitecture
from tcam_model import TCAM
from baseline import GRUHyperModel, LSTMHyperModel, CNNHyperModel, TransformerHyperModel


def save_data(X, y, patient_ids, filename):
    with open(filename, 'wb') as f:
        pickle.dump((X, y, patient_ids), f)


def load_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def initialize_base_model(hp):
    tcam_model = TCAM.from_hyperparams(hp)
    return tcam_model


def initialize_custom_model():
    model_builder = CustomModelArchitecture(input_shape=(1440, 1))
    return model_builder.build()


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


def load_and_preprocess_data(file_path, label):
    df = pd.read_csv(file_path)
    df[['Phone timestamp', 'RR-interval [ms]']] = df['Phone timestamp;RR-interval [ms]'].str.split(';', expand=True)
    df['RR-interval [ms]'] = pd.to_numeric(df['RR-interval [ms]'])
    df.drop(columns=['Phone timestamp;RR-interval [ms]'], inplace=True)
    df['label'] = label
    patient_id = file_path.split('/')[-1].split('_')[-1].split('.')[0] + '_' + str(label)
    df['patient_id'] = patient_id
    return df


def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    patient_ids = []
    for i in range(len(data) - sequence_length):
        seq = data['RR-interval [ms]'].values[i:i + sequence_length]
        label = data['label'].values[i + sequence_length - 1]
        patient_id = data['patient_id'].values[i + sequence_length - 1]
        sequences.append(seq)
        labels.append(label)
        patient_ids.append(patient_id)
    return np.array(sequences), np.array(labels), np.array(patient_ids)


def normalize_sequences(X, sequence_length):
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, 1)).reshape(-1, sequence_length, 1)
    return X


def patient_split(X, y, patient_ids, train_patient_ids, val_patient_ids):
    train_idx = np.isin(patient_ids, train_patient_ids)
    val_idx = np.isin(patient_ids, val_patient_ids)
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def get_callbacks():
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    return [early_stopping, reduce_lr]


def get_kfold(n_splits=5, random_state=41):
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
