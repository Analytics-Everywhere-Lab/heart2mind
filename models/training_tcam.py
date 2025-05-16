import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf
from keras_tuner import RandomSearch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from models.tcam_model import TCAM
from utils import get_callbacks, get_kfold
from dataloader import prepare_data, save_data, load_data, patient_split


def train_model(X, y, patient_ids, tuner, callbacks):
    unique_patient_ids = np.unique(patient_ids)
    kfold = get_kfold()
    best_model = None
    best_score = 0
    for train_patient_ids, val_patient_ids in tqdm(
        kfold.split(unique_patient_ids),
        total=kfold.get_n_splits(),
        desc="K-Fold",
    ):
        train_patient_ids = unique_patient_ids[train_patient_ids]
        val_patient_ids = unique_patient_ids[val_patient_ids]
        X_train, X_val, y_train, y_val = patient_split(
            X, y, patient_ids, train_patient_ids, val_patient_ids
        )

        X_train_tuner, X_tuner_val, y_train_tuner, y_tuner_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        tuner.search(
            X_train_tuner,
            y_train_tuner,
            epochs=100,
            validation_data=(X_tuner_val, y_tuner_val),
            callbacks=callbacks,
        )

    top_model = tuner.get_best_models(num_models=5)[0]
    top_model.save("tcam_with_mine.h5")
    return top_model


def main():
    print(
        "################################ Training: 6.1 Seed 41 #################################"
    )
    print(
        "Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU"))
    )

    data_dir = "HRV_anonymized_data"
    control_pattern = re.compile(r"control_\d+\.csv")
    treatment_pattern = re.compile(r"treatment_\d+\.csv")
    sequence_length = 50

    X, y, patient_ids = prepare_data(
        data_dir, control_pattern, treatment_pattern, sequence_length
    )
    save_data(X, y, patient_ids, "data.pkl")

    X, y, patient_ids = load_data("data.pkl")

    hypermodel = TCAM()
    tuner = RandomSearch(
        hypermodel,
        objective="val_accuracy",
        max_trials=2000,
        executions_per_trial=1,
        directory="tuner_results",
        project_name="hrv_tuning",
        max_consecutive_failed_trials=2000,
    )

    callbacks = get_callbacks()
    train_model(X, y, patient_ids, tuner, callbacks)
    print("Training complete")


if __name__ == "__main__":
    main()
