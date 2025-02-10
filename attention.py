import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid pre-allocating all the GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Optionally, you can set a memory limit
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
    except RuntimeError as e:
        print(e)

LAYER_NAME = 'multi_head_attention'
SEQUENCE_LENGTH = 50
data_dir = 'HRV_anonymized_data'
explanation_dir = 'explanations'


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def extract_attention_weights(model, inputs, layer_name=LAYER_NAME):
    layer = model.get_layer(layer_name)
    attention_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
    attention_weights = attention_model.predict(inputs)
    # Average across heads if needed
    attention_weights = np.mean(attention_weights, axis=-1)
    # Normalize to [0, 1]
    attention_weights = (attention_weights - np.min(attention_weights)) / (
            np.max(attention_weights) - np.min(attention_weights))
    return attention_weights


def overlay_attention(data, attention_weights, gt, pred_index, file_name):
    fig, ax1 = plt.subplots(figsize=(20, 8))

    # # Reverse the attention weights
    # if pred_index == 0:
    #     attention_weights = 1 - attention_weights

    extent = [0, len(data), 0, 1]
    ax1.imshow(attention_weights.T, cmap='coolwarm', aspect='auto', alpha=0.8, extent=extent)
    ax1.set_yticks([])

    # Plotting the RR-interval data
    ax2 = ax1.twinx()
    ax2.plot(data, alpha=1, color='b', label='RR-interval [ms]', linewidth=0.5)
    ax2.set_ylabel('RR-interval [ms]', fontsize=16)
    ax2.yaxis.set_label_position('left')
    ax2.tick_params(axis='y', labelsize=16)
    ax2.tick_params(axis='x', labelsize=16)

    # Show colorbar
    plt.colorbar(ax1.imshow(attention_weights.T, cmap='coolwarm', aspect='auto', alpha=0.8, extent=extent))

    plt.title('RR-interval [ms] with Attention Overlay (Ground Truth: {}, Predicted: {})'.format(gt, pred_index),
              fontsize=10)
    plt.savefig(os.path.join(explanation_dir, f'attention_{file_name}.png'))
    plt.close(fig)


def gradcam(model, x, layer_name=LAYER_NAME):
    layer = model.get_layer(layer_name)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    intermediate_model = tf.keras.models.Model(inputs=model.inputs, outputs=[layer.output, model.output])

    # Process data in smaller batches
    batch_size = 1024  # Set an appropriate batch size to avoid memory issues
    num_batches = int(np.ceil(x.shape[0] / batch_size))

    all_layer_outputs = []
    all_grads = []
    pred_indices = []

    for batch_idx in tqdm(range(num_batches)):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, x.shape[0])
        batch_x = x[batch_start:batch_end]

        with tf.GradientTape() as tape:
            tape.watch(batch_x)
            layer_output, predictions = intermediate_model(batch_x)
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, layer_output)
        all_layer_outputs.append(layer_output.numpy())
        all_grads.append(grads.numpy())
        pred_indices.append(pred_index.numpy())

    # Concatenate all collected data
    all_layer_outputs = np.concatenate(all_layer_outputs, axis=0)
    all_grads = np.concatenate(all_grads, axis=0)

    # Perform normalization
    pooled_grads = np.mean(all_grads, axis=(0, 1))
    heatmap = np.mean(np.multiply(pooled_grads, all_layer_outputs), axis=-1)
    # heatmap = np.maximum(heatmap, 0) / np.mean(heatmap)

    # Normalize the final heatmap to [0, 1]
    final_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    # final_pred_index = pred_indices[0]  # Assuming all predictions are the same for simplicity

    return final_heatmap


def overlay_gradcam(data, heatmap, gt, pred_index, file_name):
    fig, ax1 = plt.subplots(figsize=(20, 8))

    # Normalize heatmap
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

    extent = [0, len(data), 0, 1]
    ax1.imshow(heatmap.T, cmap='coolwarm', aspect='auto', alpha=0.8, extent=extent)
    ax1.set_yticks([])

    # Plotting the RR-interval data
    ax2 = ax1.twinx()
    ax2.plot(data, alpha=1, color='b', label='RR-interval [ms]', linewidth=0.5)
    ax2.set_ylabel('RR-interval [ms]', fontsize=16)
    ax2.yaxis.set_label_position('left')
    ax2.tick_params(axis='y', labelsize=16)
    # Set size of x-axis labels
    ax2.tick_params(axis='x', labelsize=16)

    # Show colorbar
    plt.colorbar(ax1.imshow(heatmap.T, cmap='coolwarm', aspect='auto', alpha=0.8, extent=extent))

    plt.title('RR-interval [ms] with Grad-CAM Overlay (Ground Truth: {}, Predicted: {})'.format(gt, pred_index),
              fontsize=10)
    plt.savefig(os.path.join(explanation_dir, f'gradcam_{file_name}.png'))
    plt.close(fig)


def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df[['Phone timestamp', 'RR-interval [ms]']] = df['Phone timestamp;RR-interval [ms]'].str.split(';', expand=True)
    df['RR-interval [ms]'] = pd.to_numeric(df['RR-interval [ms]'])
    df.drop(columns=['Phone timestamp;RR-interval [ms]'], inplace=True)
    return df


def normalize_data(data, scaler=None):
    if not scaler:
        scaler = StandardScaler()
    data = data.reshape(-1, 1)
    normalized_data = scaler.fit_transform(data)
    # Split data into sequences of SEQUENCE_LENGTH
    sequences = []
    for i in range(len(normalized_data) - SEQUENCE_LENGTH + 1):
        sequences.append(normalized_data[i:i + SEQUENCE_LENGTH])
    return np.array(sequences), scaler


def main():
    model_path = 'best_model_separate_2.h5'
    # Load model
    model = load_model(model_path)
    # Regular expressions to match control and treatment files
    control_pattern = re.compile(r'control_\d+\.csv')
    treatment_pattern = re.compile(r'treatment_\d+\.csv')

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            test_file_path = os.path.join(data_dir, file_name)
            if control_pattern.match(file_name):
                print(f"Ground truth label for {file_name}: 0")
                gt = 0
            elif treatment_pattern.match(file_name):
                print(f"Ground truth label for {file_name}: 1")
                gt = 1

            print(test_file_path)

            # Preprocess and normalize data
            df = preprocess_data(test_file_path)
            X_test, scaler = normalize_data(df['RR-interval [ms]'].values)

            # Predicted class index
            y_pred = model.predict(X_test)
            real_pred_index = np.argmax(y_pred, axis=-1)[0]
            print("Predicted class index:", real_pred_index)

            # Extract and overlay attention weights
            attention_weights = extract_attention_weights(model, X_test)
            overlay_attention(df['RR-interval [ms]'].values, attention_weights, gt, real_pred_index, file_name)

            # Generate and overlay Grad-CAM heatmap
            heatmap = gradcam(model, X_test)
            overlay_gradcam(df['RR-interval [ms]'].values, heatmap, gt, real_pred_index, file_name)


if __name__ == "__main__":
    main()
