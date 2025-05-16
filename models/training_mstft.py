import re
from sklearn.model_selection import KFold 
import numpy as np
import tensorflow as tf
import json
from dataloader import load_data, patient_split, prepare_data, save_data
from models.mstft import MSTFT
from utils import get_callbacks, PrintMetricsCallback

# Suppress tensorflow warnings
tf.get_logger().setLevel("ERROR")

def train_model(X, y, patient_ids, callbacks, sequence_length):
    best_model = None
    results_log = []

    unique_patient_ids = np.unique(patient_ids)
    kfold = KFold(n_splits=5, shuffle=True, random_state=41)

    for i, (train_patient_ids, val_patient_ids) in enumerate(
        kfold.split(unique_patient_ids)
    ):
        print(f"Fold {i+1}")
        train_patient_ids = unique_patient_ids[train_patient_ids]
        val_patient_ids = unique_patient_ids[val_patient_ids]
        X_train, X_val, y_train, y_val = patient_split(
            X, y, patient_ids, train_patient_ids, val_patient_ids
        )

        model_builder = MSTFT(sequence_length)
        model = model_builder.build()
        
        y_train = y_train.reshape(-1, 1)  # Shape (n_samples, 1)
        y_val = y_val.reshape(-1, 1)
        model.fit(
            X_train,
            y_train,
            batch_size=32,
            validation_data=(X_val, y_val),
            epochs=1000,
            verbose=1,
            callbacks=callbacks,
        )
        train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = model.evaluate(
            X_train, y_train, verbose=0
        )
        print(
            f"\nFold Train Results -> "
            f"loss={train_loss:.4f}, acc={train_acc:.4f}, prec={train_prec:.4f}, "
            f"rec={train_rec:.4f}, f1={train_f1:.4f}, auc={train_auc:.4f}"
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = model.evaluate(
            X_val, y_val, verbose=0
        )
        print(
            f"\nFold Val Results -> "
            f"loss={val_loss:.4f}, acc={val_acc:.4f}, prec={val_prec:.4f}, "
            f"rec={val_rec:.4f}, f1={val_f1:.4f}, auc={val_auc:.4f}"
        )
        # Append results to log
        fold_results = {
            "fold": i + 1,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "train_prec": float(train_prec),
            "train_recall": float(train_rec),
            "train_f1": float(train_f1),
            "train_auc": float(train_auc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_prec": float(val_prec),
            "val_recall": float(val_rec),
            "val_f1": float(val_f1),
            "val_auc": float(val_auc),
        }
        results_log.append(fold_results)

        model.save(f"best_mstft_final.h5")
        with open("results_log.json", "w") as f:
            json.dump(results_log, f, indent=2)
    return best_model


def _save_data(sequence_length):
    data_dir = "HRV_anonymized_data"
    control_pattern = re.compile(r"control_\d+\.csv")
    treatment_pattern = re.compile(r"treatment_\d+\.csv")

    X, y, patient_ids = prepare_data(
        data_dir, control_pattern, treatment_pattern, sequence_length
    )
    save_data(X, y, patient_ids, f"data_{sequence_length}.pkl")


def main():
    sequence_length = 1000
    _save_data(sequence_length)
    X, y, patient_ids = load_data(f"data_{sequence_length}.pkl")
    callbacks = get_callbacks(patience=90)
    train_model(X, y, patient_ids, callbacks, sequence_length)


if __name__ == "__main__":
    main()
