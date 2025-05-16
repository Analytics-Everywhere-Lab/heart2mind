import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score, \
    roc_curve, auc
from sklearn.model_selection import LeaveOneOut
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm

from hyperparams_set import *
from utils import *

# Hide warnings
import warnings

warnings.filterwarnings('ignore')


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def evaluate_model(model, X, y, patient_ids):
    unique_patient_ids = np.unique(patient_ids)
    loo = LeaveOneOut()

    initial_weights = model.get_weights()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    # Initialize lists to store aggregated patient-level predictions and true labels
    all_patient_true_labels = []
    all_patient_pred_labels = []
    all_patient_pred_probs = []

    for fold, (train_idx, val_idx) in enumerate(
            tqdm(loo.split(unique_patient_ids), total=len(unique_patient_ids), desc="LOO Progress")):
        model.set_weights(initial_weights)

        val_patient_ids = unique_patient_ids[val_idx]
        train_patient_ids = unique_patient_ids[train_idx]

        # Get indices of samples for training and validation patients
        train_sample_idx = np.isin(patient_ids, train_patient_ids)
        val_sample_idx = np.isin(patient_ids, val_patient_ids)

        X_train_fold = X[train_sample_idx]
        X_val_fold = X[val_sample_idx]
        y_train_fold = y[train_sample_idx]
        y_val_fold = y[val_sample_idx]

        model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=3, verbose=1,
                  callbacks=[early_stopping, reduce_lr])

        y_pred_prob = model.predict(X_val_fold)
        y_pred_classes = np.argmax(y_pred_prob, axis=1)

        # Aggregate predictions for the validation patient
        # Since val_patient_ids contains only one patient, val_patient_ids[0]
        patient_id = val_patient_ids[0]

        # Aggregate sample-level predictions to patient-level prediction
        # For predicted class, take the mode
        patient_pred_class = np.bincount(y_pred_classes).argmax()

        # For predicted probability, take the average probability for class 1
        patient_pred_prob = np.mean(y_pred_prob[:, 1])  # Assuming class 1 is the positive class

        # Get the true label for this patient
        patient_true_label = y_val_fold[0]  # Assuming all samples for this patient have the same label

        # Append to the lists
        all_patient_true_labels.append(patient_true_label)
        all_patient_pred_labels.append(patient_pred_class)
        all_patient_pred_probs.append(patient_pred_prob)

    # After all folds, compute metrics over patients
    precision = precision_score(all_patient_true_labels, all_patient_pred_labels, zero_division=0)
    recall = recall_score(all_patient_true_labels, all_patient_pred_labels, zero_division=0)
    f1 = f1_score(all_patient_true_labels, all_patient_pred_labels, zero_division=0)
    accuracy = accuracy_score(all_patient_true_labels, all_patient_pred_labels)
    confusion_mat = confusion_matrix(all_patient_true_labels, all_patient_pred_labels)

    # ROC AUC
    roc_auc = roc_auc_score(all_patient_true_labels, all_patient_pred_probs)

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_patient_true_labels, all_patient_pred_probs)
    roc_auc_curve = auc(fpr, tpr)

    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'confusion_matrix': confusion_mat,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc_curve': roc_auc_curve
    }

    return results


def save_results(results, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)


def load_results(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def plot_results(results, suffix):
    print(f'Precision: {results["precision"]}')
    print(f'Recall: {results["recall"]}')
    print(f'F1 Score: {results["f1"]}')
    print(f'ROC AUC: {results["roc_auc"]}')
    print(f'Accuracy: {results["accuracy"]}')
    print(f'Confusion Matrix:\n{results["confusion_matrix"]}')

    # Plot ROC Curve
    plt.figure()
    plt.plot(results['fpr'], results['tpr'], label=f'ROC curve (area = {results["roc_auc_curve"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic {suffix}')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_{suffix}.pdf')
    plt.show()

    # Plot Confusion Matrix
    plt.figure()
    plot_confusion_matrix(results['confusion_matrix'], classes=['Control', 'Treatment'], normalize=True,
                          title=f'Confusion Matrix {suffix}')
    plt.savefig(f'confusion_matrix_{suffix}.pdf')
    plt.show()


def main():
    X, y, patient_ids = load_data('data_1440.pkl')

    # hyperparameter_sets = [hyper.hyperparameters_2]

    # for i, hyperparams in enumerate(hyperparameter_sets):
    #     print(f"Evaluating hyperparameter set {i}: {hyperparams}")
    # hyperparams_set = cnn_hyperparameters_0
    #
    # # model = initialize_custom_model()
    # model = initialize_CNN_model(hyperparams_set)
    # results = evaluate_model(model, X, y, patient_ids)
    # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # save_results(results, f'evaluation_results_loocv_cnn_{timestamp}.pkl')
    # # plot_results(results, suffix=f'evaluation_results_loocv_cnn_{timestamp}')
    #
    # hyperparams_set = gru_hyperparameters_0
    #
    # # model = initialize_custom_model()
    # model = initialize_GRU_model(hyperparams_set)
    # results = evaluate_model(model, X, y, patient_ids)
    # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # save_results(results, f'evaluation_results_loocv_gru_{timestamp}.pkl')
    # # plot_results(results, suffix=f'evaluation_results_loocv_gru_{timestamp}')
    #
    # hyperparams_set = lstm_hyperparameters_0
    #
    # # model = initialize_custom_model()
    # model = initialize_LSTM_model(hyperparams_set)
    # results = evaluate_model(model, X, y, patient_ids)
    # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # save_results(results, f'evaluation_results_loocv_lstm_{timestamp}.pkl')
    #
    # hyperparams_set = lstm_hyperparameters_0

    # model = initialize_custom_model()
    model = initialize_custom_model()
    results = evaluate_model(model, X, y, patient_ids)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_results(results, f'evaluation_results_loocv_misgar_{timestamp}.pkl')



if __name__ == "__main__":
    main()
