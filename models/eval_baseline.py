import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
# Hide warnings
import warnings
from datetime import datetime
from kerastuner import RandomSearch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, roc_curve, auc
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from baseline import LSTMHyperModel, GRUHyperModel, CNNHyperModel, TransformerHyperModel
from utils import load_data, get_callbacks, get_kfold, patient_split

warnings.filterwarnings('ignore')

plt.tight_layout()


def plot_roc_curve(fpr, tpr, roc_auc, fold):
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC fold {fold + 1} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Fold {fold + 1}')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_fold_{fold + 1}.pdf')
    plt.close()


def tune_hyperparameters(hypermodel, X_train, y_train, X_val, y_val):
    tuner = RandomSearch(
        hypermodel,
        objective='val_accuracy',
        max_trials=1,
        executions_per_trial=1,
        directory='tuner_results',
        project_name=f'{hypermodel.__class__.__name__}_tuning',
        max_consecutive_failed_trials=1
    )

    callbacks = get_callbacks()
    tuner.search(X_train, y_train, epochs=1, validation_data=(X_val, y_val), callbacks=callbacks)
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hyperparameters


def build_model_from_hyperparameters(hypermodel_class, best_hyperparameters):
    hypermodel = hypermodel_class()
    model = hypermodel.build(best_hyperparameters)
    return model


def evaluate_model(model, X, y, patient_ids, n_splits=5):
    unique_patient_ids = np.unique(patient_ids)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=41)

    callbacks = get_callbacks()

    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []
    accuracy_scores = []
    fpr_list = []
    tpr_list = []
    roc_auc_list = []

    for fold, (train_idx, val_idx) in enumerate(
            tqdm(kfold.split(unique_patient_ids), total=kfold.get_n_splits(), desc="K-Fold Progress")):
        val_patient_ids = unique_patient_ids[val_idx]
        train_patient_ids = unique_patient_ids[train_idx]

        X_train_fold = X[np.isin(patient_ids, train_patient_ids)]
        X_val_fold = X[np.isin(patient_ids, val_patient_ids)]
        y_train_fold = y[np.isin(patient_ids, train_patient_ids)]
        y_val_fold = y[np.isin(patient_ids, val_patient_ids)]

        model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=100, verbose=1,
                  callbacks=callbacks)

        y_pred_prob = model.predict(X_val_fold)
        y_pred_classes = np.argmax(y_pred_prob, axis=1)

        val_patient_indices = np.where(np.isin(patient_ids, val_patient_ids))[0]
        aggregated_predictions = []
        aggregated_labels = []
        aggregated_probabilities = []
        for patient_id in val_patient_ids:
            patient_indices = np.where(patient_ids[val_patient_indices] == patient_id)[0]
            patient_sequences = y_pred_classes[patient_indices]
            patient_label = y[val_patient_indices][patient_indices][0]
            most_common_prediction = np.bincount(patient_sequences).argmax()
            aggregated_predictions.append(most_common_prediction)
            aggregated_labels.append(patient_label)
            avg_probability = np.mean(y_pred_prob[patient_indices, 1])
            aggregated_probabilities.append(avg_probability)

        precision_scores.append(precision_score(aggregated_labels, aggregated_predictions))
        recall_scores.append(recall_score(aggregated_labels, aggregated_predictions))
        f1_scores.append(f1_score(aggregated_labels, aggregated_predictions))
        roc_auc_scores.append(roc_auc_score(aggregated_labels, aggregated_probabilities))
        accuracy_scores.append(accuracy_score(aggregated_labels, aggregated_predictions))
        fpr, tpr, _ = roc_curve(aggregated_labels, aggregated_probabilities)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(auc(fpr, tpr))

        # Plot ROC curve for this fold
        plot_roc_curve(fpr, tpr, roc_auc_list[-1], fold)

    return {
        'precision_scores': precision_scores,
        'recall_scores': recall_scores,
        'f1_scores': f1_scores,
        'roc_auc_scores': roc_auc_scores,
        'accuracy_scores': accuracy_scores,
        'fpr_list': fpr_list,
        'tpr_list': tpr_list,
        'roc_auc_list': roc_auc_list
    }


def save_results(results, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)


def load_results(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def plot_results(results, suffix):
    mean_precision = np.mean(results['precision_scores'])
    mean_recall = np.mean(results['recall_scores'])
    mean_f1 = np.mean(results['f1_scores'])
    mean_roc_auc = np.mean(results['roc_auc_scores']) if results['roc_auc_scores'] else 'undefined'
    mean_accuracy = np.mean(results['accuracy_scores'])

    print(f'Mean Precision: {mean_precision}')
    print(f'Mean Recall: {mean_recall}')
    print(f'Mean F1 Score: {mean_f1}')
    print(f'Mean ROC AUC: {mean_roc_auc}')
    print(f'Mean Accuracy: {mean_accuracy}')

    df_scores = pd.DataFrame({
        'Precision': results['precision_scores'],
        'Recall': results['recall_scores'],
        'F1 Score': results['f1_scores'],
        'ROC AUC': results['roc_auc_scores'],
        'Accuracy': results['accuracy_scores']
    })

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_scores)
    plt.title(f'Evaluation Metrics')
    plt.ylabel('Score')
    plt.xlabel('Metrics')
    plt.savefig(f'evaluation_metrics_{suffix}.pdf')
    plt.close()

    if results['roc_auc_list']:
        # Plot ROC curve for all folds
        plt.figure(figsize=(10, 6))
        for fold, (fpr, tpr) in enumerate(zip(results['fpr_list'], results['tpr_list'])):
            plt.plot(fpr, tpr, label=f'ROC fold {fold + 1} (AUC = {results["roc_auc_list"][fold]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves for All Folds {suffix}')
        plt.legend(loc="lower right")
        plt.savefig(f'roc_curves_all_{suffix}.pdf')
        plt.close()

def train_model(X, y, patient_ids, tuner, callbacks, model_name):
    unique_patient_ids = np.unique(patient_ids)
    kfold = get_kfold()
    for train_patient_ids, val_patient_ids in tqdm(kfold.split(unique_patient_ids), total=kfold.get_n_splits(),
                                                   desc="K-Fold Progress"):
        train_patient_ids = unique_patient_ids[train_patient_ids]
        val_patient_ids = unique_patient_ids[val_patient_ids]
        X_train, X_val, y_train, y_val = patient_split(X, y, patient_ids, train_patient_ids, val_patient_ids)

        X_train_tuner, X_tuner_val, y_train_tuner, y_tuner_val = train_test_split(X_train,
                                                                                  y_train,
                                                                                  test_size=0.2,
                                                                                  random_state=42)

        tuner.search(X_train_tuner, y_train_tuner, epochs=10, validation_data=(X_tuner_val, y_tuner_val),
                     callbacks=callbacks)

    top_model = tuner.get_best_models(num_models=5)[0]
    top_model.save(f'best_model_{model_name}.h5')
    return top_model

def main():
    X, y, patient_ids = load_data('data.pkl')
    # Tune hyperparameters [LSTMHyperModel(), GRUHyperModel(), TransformerHyperModel(), CNNHyperModel()]
    hypermodels =  [TransformerHyperModel()]
    best_hyperparameters_dict = {}

    for hypermodel in hypermodels:
        model_name = hypermodel.__class__.__name__
        tuner = RandomSearch(
            hypermodel,
            objective='val_accuracy',
            max_trials=10,
            executions_per_trial=1,
            directory=f'tuner_results/{model_name}',
            project_name='hrv_tuning',
            max_consecutive_failed_trials=10
        )

        print(f"Tuning hyperparameters for {model_name}")
        callbacks = get_callbacks()
        train_model(X, y, patient_ids, tuner, callbacks, model_name)
        print("Tuning completed")



if __name__ == "__main__":
    main()
