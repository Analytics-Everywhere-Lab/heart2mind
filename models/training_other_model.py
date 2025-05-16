import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from model import CustomModelArchitecture
from utils import get_callbacks, get_kfold
from dataloader import prepare_data, save_data, load_data, patient_split


def train_custom_model(X, y, patient_ids, callbacks, sequence_length):
    unique_patient_ids = np.unique(patient_ids)
    kfold = get_kfold()
    best_model = None
    best_score = 0
    for train_patient_ids, val_patient_ids in tqdm(
        kfold.split(unique_patient_ids),
        total=kfold.get_n_splits(),
        desc="K-Fold Progress",
    ):
        train_patient_ids = unique_patient_ids[train_patient_ids]
        val_patient_ids = unique_patient_ids[val_patient_ids]
        X_train, X_val, y_train, y_val = patient_split(
            X, y, patient_ids, train_patient_ids, val_patient_ids
        )
        model_builder = CustomModelArchitecture(input_shape=(sequence_length, 1))
        model = model_builder.build()
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            verbose=1,
            callbacks=callbacks,
        )
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(
            f"Validation accuracy: {val_accuracy:.4f}",
            f"Validation loss: {val_loss:.4f}",
        )
        model.save(f"best_model_custom_{best_score:.4f}.h5")
    return best_model


def main():
    print(
        "################################ Training Custom Model #################################"
    )
    print(
        "Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU"))
    )

    data_dir = "HRV_anonymized_data"
    control_pattern = re.compile(r"control_\d+\.csv")
    treatment_pattern = re.compile(r"treatment_\d+\.csv")
    sequence_length = 1440

    X, y, patient_ids = prepare_data(
        data_dir, control_pattern, treatment_pattern, sequence_length
    )
    save_data(X, y, patient_ids, "data_1440.pkl")

    X, y, patient_ids = load_data("data_1440.pkl")

    callbacks = get_callbacks()
    train_custom_model(X, y, patient_ids, callbacks, sequence_length)
    print("Training complete")


if __name__ == "__main__":
    main()
