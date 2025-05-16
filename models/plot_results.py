import pickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization

print("################################ Plot Result: 1.0 #################################")

model_paths = ['best_model_LSTMHyperModel.h5',
               'best_model_GRUHyperModel.h5',
               'best_model_CNNHyperModel.h5',
               'best_model_TransformerHyperModel.h5']
model_names = [
    'LSTM',
    'GRU',
    '1D-CNN',
    'Transformer']
data_files = ['data.pkl', 'data.pkl', 'data.pkl', 'data.pkl']


# Load the best model with custom object scope
def load_model_with_custom_objects(model_path):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"InstanceNormalization": InstanceNormalization}
    )


# Function to evaluate a model
def evaluate_model(model, X, y, model_name, patient_ids):
    initial_weights = model.get_weights()
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []
    roc_auc_scores = []
    roc_data = []

    # Define the callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(
            tqdm(kfold.split(X), total=kfold.get_n_splits(), desc=f"K-Fold Progress for {model_name}")):
        print(f'Fold {fold + 1} started for {model_name}')
        model.set_weights(initial_weights)

        unique_patient_ids = np.unique(patient_ids)
        train_patient_ids, val_patient_ids = train_test_split(unique_patient_ids, test_size=0.2, random_state=2)

        X_train = X[np.isin(patient_ids, train_patient_ids)]
        X_val = X[np.isin(patient_ids, val_patient_ids)]
        y_train = y[np.isin(patient_ids, train_patient_ids)]
        y_val = y[np.isin(patient_ids, val_patient_ids)]

        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, verbose=1, callbacks=callbacks)

        y_pred_prob = model.predict(X_val)
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

        precision = precision_score(aggregated_labels, aggregated_predictions)
        recall = recall_score(aggregated_labels, aggregated_predictions)
        f1 = f1_score(aggregated_labels, aggregated_predictions)
        roc_auc = roc_auc_score(aggregated_labels, aggregated_probabilities)
        accuracy = accuracy_score(aggregated_labels, aggregated_predictions)
        fpr, tpr, _ = roc_curve(aggregated_labels, aggregated_probabilities)
        roc_data.append((fpr, tpr, roc_auc))

    return precision, recall, f1, accuracy, roc_auc, roc_data


def main():
    # Initialize results storage
    all_results = {
        name: {'precision_scores': [], 'recall_scores': [], 'f1_scores': [], 'roc_auc_scores': [], 'roc_data': []} for
        name in model_names}

    # Evaluate each model
    for model_path, model_name, data_file in zip(model_paths, model_names, data_files):
        # Load the corresponding data
        with open(data_file, 'rb') as f:
            X, y, patient_ids = pickle.load(f)
        model = load_model_with_custom_objects(model_path)
        precision_scores, recall_scores, f1_scores, accuracy_scores, roc_auc_scores, roc_data = evaluate_model(model, X,
                                                                                                               y,
                                                                                                               model_name,
                                                                                                               patient_ids)

        all_results[model_name]['precision_scores'] = precision_scores
        all_results[model_name]['recall_scores'] = recall_scores
        all_results[model_name]['f1_scores'] = f1_scores
        all_results[model_name]['accuracy_scores'] = accuracy_scores
        all_results[model_name]['roc_auc_scores'] = roc_auc_scores
        all_results[model_name]['roc_data'] = roc_data

    # Save results
    with open('all_model_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    # Load results
    with open('all_model_results.pkl', 'rb') as f:
        all_results = pickle.load(f)

    # Display results
    for model_name in model_names:
        precision_scores = all_results[model_name]['precision_scores']
        recall_scores = all_results[model_name]['recall_scores']
        f1_scores = all_results[model_name]['f1_scores']
        accuracy_scores = all_results[model_name]['accuracy_scores']
        roc_auc_scores = all_results[model_name]['roc_auc_scores']

        print(f"Results for {model_name}:")
        print("Precision scores:", precision_scores)
        print("Recall scores:", recall_scores)
        print("F1 scores:", f1_scores)
        print("Accuracy scores:", accuracy_scores)
        print("ROC AUC scores:", roc_auc_scores)
        print("Average Precision:", np.mean(precision_scores))
        print("Average Recall:", np.mean(recall_scores))
        print("Average F1 Score:", np.mean(f1_scores))
        print("Average Accuracy:", np.mean(accuracy_scores))
        print("Average ROC AUC:", np.mean(roc_auc_scores))

    # Plot ROC Curves with mean and standard deviation
    plt.figure(figsize=(10, 6))
    for model_name in model_names:
        roc_data = all_results[model_name]['roc_data']
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        for fpr, tpr, _ in roc_data:
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std([roc_auc for _, _, roc_auc in roc_data])

        plt.plot(mean_fpr, mean_tpr, label=f'{model_name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
        plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves with Mean and Standard Deviation')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves_all_models.pdf')
    plt.close()


if __name__ == '__main__':
    main()
