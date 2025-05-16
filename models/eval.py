import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm

import hyper  # Import the hyperparameters
from utils import initialize_base_model, load_data, initialize_custom_model

# Hide warnings
import warnings
warnings.filterwarnings('ignore')

plt.tight_layout()

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

def evaluate_model(model, X, y, patient_ids, n_splits=5):
    unique_patient_ids = np.unique(patient_ids)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=41)

    initial_weights = model.get_weights()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []
    accuracy_scores = []
    confusion_matrices = []
    fpr_list = []
    tpr_list = []
    roc_auc_list = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(unique_patient_ids), total=kfold.get_n_splits(), desc="K-Fold Progress")):
        model.set_weights(initial_weights)

        val_patient_ids = unique_patient_ids[val_idx]
        train_patient_ids = unique_patient_ids[train_idx]

        X_train_fold = X[np.isin(patient_ids, train_patient_ids)]
        X_val_fold = X[np.isin(patient_ids, val_patient_ids)]
        y_train_fold = y[np.isin(patient_ids, train_patient_ids)]
        y_val_fold = y[np.isin(patient_ids, val_patient_ids)]

        model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=100, verbose=1, callbacks=[early_stopping, reduce_lr])

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
        confusion_matrices.append(confusion_matrix(aggregated_labels, aggregated_predictions))
        fpr, tpr, _ = roc_curve(aggregated_labels, aggregated_probabilities)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(auc(fpr, tpr))

        # Plot ROC curve for this fold
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC fold {fold + 1} (AUC = {roc_auc_list[-1]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Fold {fold + 1}')
        plt.legend(loc="lower right")
        plt.savefig(f'roc_curve_fold_{fold + 1}.pdf')
        plt.close()

    return {
        'precision_scores': precision_scores,
        'recall_scores': recall_scores,
        'f1_scores': f1_scores,
        'roc_auc_scores': roc_auc_scores,
        'accuracy_scores': accuracy_scores,
        'confusion_matrices': confusion_matrices,
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
    mean_roc_auc = np.mean(results['roc_auc_scores'])
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

    mean_confusion_matrix = np.mean(results['confusion_matrices'], axis=0)
    class_names = ['Control', 'Treatment']
    plt.figure()
    plot_confusion_matrix(mean_confusion_matrix, classes=class_names, normalize=True, title=f'Mean Confusion Matrix {suffix}')
    plt.savefig(f'confusion_matrix_{suffix}.pdf')
    plt.close()

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

def main():
    X, y, patient_ids = load_data('data.pkl')

    hyperparameter_sets = [hyper.hyperparameters_2]

    for i, hyperparams in enumerate(hyperparameter_sets):
        print(f"Evaluating hyperparameter set {i}: {hyperparams}")
        model = initialize_base_model(hyperparams)
        results = evaluate_model(model, X, y, patient_ids)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = f"final_{timestamp}"
    #     save_results(results, f'evaluation_results_{suffix}.pkl')
    with open('evaluation_results_2.pkl', 'rb') as f:
        results = pickle.load(f)
    plot_results(results, suffix=suffix)

    # ==== Custom Model ====
    # model = initialize_custom_model()
    # results = evaluate_model(model, X, y, patient_ids)
    # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # suffix = f"custom_{timestamp}"
    # save_results(results, f'evaluation_results_{suffix}.pkl')
    # plot_results(results, suffix=suffix)

if __name__ == "__main__":
    main()
