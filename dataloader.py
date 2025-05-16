from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import pickle
import tensorflow_addons as tfa
import tensorflow as tf

def save_data(X, y, patient_ids, filename):
    with open(filename, "wb") as f:
        pickle.dump((X, y, patient_ids), f)


def load_data(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_and_preprocess_data(file_path, label):
    df = pd.read_csv(file_path)
    df[["Phone timestamp", "RR-interval [ms]"]] = df[
        "Phone timestamp;RR-interval [ms]"
    ].str.split(";", expand=True)
    df["RR-interval [ms]"] = pd.to_numeric(df["RR-interval [ms]"])
    df.drop(columns=["Phone timestamp;RR-interval [ms]"], inplace=True)
    df["label"] = label
    patient_id = (
        file_path.split("/")[-1].split("_")[-1].split(".")[0] + "_" + str(label)
    )
    df["patient_id"] = patient_id
    return df


def prepare_data(data_dir, control_pattern, treatment_pattern, sequence_length):
    dataframes = []
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if control_pattern.match(file_name):
            dataframes.append(load_and_preprocess_data(file_path, label=0))
        elif treatment_pattern.match(file_name):
            dataframes.append(load_and_preprocess_data(file_path, label=1))
    combined_df = pd.concat(dataframes, ignore_index=True)
    X, y, patient_ids = create_sequences(combined_df, sequence_length)
    X = normalize_sequences(X, sequence_length)
    return X, y, patient_ids


def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    patient_ids = []
    for i in range(0, len(data) - sequence_length, 100):
        seq = data["RR-interval [ms]"].values[i : i + sequence_length]
        label = data["label"].values[i + sequence_length - 1]
        patient_id = data["patient_id"].values[i + sequence_length - 1]
        sequences.append(seq)
        labels.append(label)
        patient_ids.append(patient_id)
    return np.array(sequences), np.array(labels, dtype="uint8"), np.array(patient_ids)


def normalize_sequences(X, sequence_length):
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, 1)).reshape(-1, sequence_length, 1)
    return X


def patient_split(X, y, patient_ids, train_patient_ids, val_patient_ids):
    train_idx = np.isin(patient_ids, train_patient_ids)
    val_idx = np.isin(patient_ids, val_patient_ids)
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]