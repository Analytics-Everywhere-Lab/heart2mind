import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tcam_model import TCAM
from utils import load_and_preprocess_data, create_sequences, normalize_sequences, patient_split, save_data, \
    load_data, get_callbacks, initialize_base_model
from hyperparams_set import *


def prepare_data(data_dir, control_pattern, treatment_pattern, sequence_length):
    dataframes = []
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if control_pattern.match(file_name):
            dataframes.append(load_and_preprocess_data(file_path, label=0))  # Control: label=0
        elif treatment_pattern.match(file_name):
            dataframes.append(load_and_preprocess_data(file_path, label=1))  # Treatment: label=1
    combined_df = pd.concat(dataframes, ignore_index=True)
    X, y, patient_ids = create_sequences(combined_df, sequence_length)
    X = normalize_sequences(X, sequence_length)
    return X, y, patient_ids


def split_train(X, y, patient_ids):
    # No validation set, just return the full training data
    return X, y, patient_ids


def train_model(X, y, patient_ids, hyperparams, callbacks):
    # Initialize the model with predefined hyperparameters
    model = initialize_base_model(hyperparams)

    # Split the data into training set (no validation)
    X_train, y_train, patient_ids_train = split_train(X, y, patient_ids)

    # Train the model on the entire dataset (no validation)
    model.fit(X_train, y_train, epochs=100, callbacks=callbacks)

    # Save the trained model
    model.save('tcam_with_mine_2.keras')
    return model


def main():
    print("################################ Training: 6.1 Seed 41 #################################")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    data_dir = 'HRV_anonymized_data'
    control_pattern = re.compile(r'control_\d+\.csv')
    treatment_pattern = re.compile(r'treatment_\d+\.csv')
    sequence_length = 50

    # Prepare data
    X, y, patient_ids = prepare_data(data_dir, control_pattern, treatment_pattern, sequence_length)
    save_data(X, y, patient_ids, 'data.pkl')

    # Load saved data
    X, y, patient_ids = load_data('data.pkl')

    # Use predefined hyperparameters (hyperparameters_2)
    hyperparams = base_hyperparameters_2  # Assuming hyperparameters_2 is a valid dictionary

    callbacks = get_callbacks()  # Adjust based on your callback functions

    # Train model with predefined hyperparameters
    trained_model = train_model(X, y, patient_ids, hyperparams, callbacks)


if __name__ == "__main__":
    main()
