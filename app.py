import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from fastdtw import fastdtw
from tensorflow_addons.layers import SpectralNormalization
import tensorflow_addons as tfa

custom_objects = {
    "SpectralNormalization": tfa.layers.SpectralNormalization,
    "GroupNormalization": tfa.layers.GroupNormalization,
    "F1Score": tfa.metrics.F1Score,
    "BinaryFocalCrossentropy": tf.keras.losses.BinaryFocalCrossentropy
}

TARGET_LAYERS = [
    "bidirectional",
    "multi_head_attention",
]
SEQUENCE_LENGTH = 50
data_dir = "HRV_anonymized_data"
explanation_dir = "explanations"

# Load the pre-trained model
# model = tf.keras.models.load_model("best_mstft_5.h5", custom_objects=custom_objects)
model = tf.keras.models.load_model("tcam_with_mine.h5")
# No grad and eval
model.trainable = False
model.compile()


def normalize_data(df):
    data = df["RR-interval [ms]"].values
    scaler = StandardScaler()
    data = data.reshape(-1, 1)
    normalized_data = scaler.fit_transform(data)
    sequences = []
    for i in range(len(normalized_data) - SEQUENCE_LENGTH + 1):
        sequences.append(normalized_data[i : i + SEQUENCE_LENGTH])
    return np.array(sequences), scaler


def predict_class(file_path):
    file_path = os.path.join(data_dir, file_path)
    df = preprocess_data(file_path)
    X_test, _ = normalize_data(df)
    y_pred = model.predict(X_test, batch_size=1024)
    predicted_class = np.argmax(y_pred, axis=-1)[0]
    # predicted_class = (y_pred > 0.5).astype(int)[0][0]
    predicted_text = "Control" if predicted_class == 0 else "Treatment"
    return predicted_text


def expand_explanations(explanation_2d, signal_length):
    """Expand 2D explanation (n_sequences, n_features) to match original signal length"""
    n_sequences, seq_len = explanation_2d.shape
    explanation_1d = explanation_2d.mean(axis=1)  # Aggregate features
    expanded = np.zeros(signal_length)
    counts = np.zeros(signal_length)

    # Distribute sequence-level explanations to corresponding time steps
    for i in range(n_sequences):
        for j in range(seq_len):
            pos = i + j
            if pos < signal_length:
                expanded[pos] += explanation_1d[i]
                counts[pos] += 1

    # Normalize by number of overlapping sequences and overall
    expanded = np.divide(expanded, counts, where=counts != 0)
    # expanded = (expanded - np.min(expanded)) / (
    #     np.max(expanded) - np.min(expanded) + 1e-8
    # )
    mean_expanded = np.mean(expanded)
    std_expanded = np.std(expanded)
    z = (expanded - mean_expanded) / (std_expanded + 1e-8)
    expanded = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-8)
    return expanded


def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df[["Phone timestamp", "RR-interval [ms]"]] = df[
        "Phone timestamp;RR-interval [ms]"
    ].str.split(";", expand=True)
    df["RR-interval [ms]"] = pd.to_numeric(df["RR-interval [ms]"])
    df.drop(columns=["Phone timestamp;RR-interval [ms]"], inplace=True)
    return df


def extract_grad_weights(model, input, signal_length):
    """
    Compute a gradient-based explanation from multiple layers and combine them.
    For each layer in EXPLANATION_LAYERS, we compute the gradient-based explanation
    (weighted by model confidence) and then average the results.
    """
    batch_size = 1024
    x = tf.convert_to_tensor(input, dtype=tf.float32)
    num_batches = int(np.ceil(x.shape[0] / batch_size))

    all_layer_explanations = []  # Will store explanation maps from each layer

    # Loop over each layer to include in the explanation.
    for layer_name in TARGET_LAYERS:
        layer_explanations = []
        for batch_idx in tqdm(range(num_batches), desc=f"Processing {layer_name}"):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, x.shape[0])
            batch_x = x[batch_start:batch_end]

            with tf.GradientTape() as tape:
                tape.watch(batch_x)
                layer = model.get_layer(layer_name)
                intermediate_model = tf.keras.models.Model(
                    inputs=model.inputs, outputs=[layer.output, model.output]
                )
                layer_output, predictions = intermediate_model(batch_x, training=False)
                pred_indices = tf.argmax(predictions, axis=-1)
                pred_indices = tf.cast(pred_indices, tf.int32)
                # confidence = tf.reduce_max(predictions, axis=-1)
                indices = tf.stack(
                    [tf.range(tf.shape(predictions)[0]), pred_indices], axis=1
                )
                losses = tf.gather_nd(predictions, indices=indices)
                loss = tf.reduce_sum(losses)

            grads = tape.gradient(loss, layer_output)
            # Compute per-example explanation for this batch:
            pooled_grads = np.mean(grads.numpy(), axis=1)  # shape: (batch, features)
            batch_explanation = np.mean(
                layer_output.numpy() * pooled_grads[:, np.newaxis, :], axis=-1
            )
            batch_explanation = np.maximum(
                batch_explanation, 0
            )  # Keep positive gradients
            # batch_explanation = batch_explanation * confidence.numpy()[:, None]

            layer_explanations.append(batch_explanation)

        # Concatenate results for this layer.
        layer_explanations = np.concatenate(layer_explanations, axis=0)
        all_layer_explanations.append(layer_explanations)

    # Combine explanations from all layers by averaging.
    combined_explanation = np.mean(np.stack(all_layer_explanations, axis=0), axis=0)
    # Normalize the combined explanation to [0, 1].
    # combined_explanation = (combined_explanation - np.min(combined_explanation)) / (
    #     np.max(combined_explanation) - np.min(combined_explanation) + 1e-8
    # )
    # mean_combined_explanation = np.mean(combined_explanation)
    # std_combined_explanation = np.std(combined_explanation)
    # z = (combined_explanation - mean_combined_explanation) / (
    #     std_combined_explanation + 1e-8
    # )
    # final_combined_explanation = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-8)
    return expand_explanations(combined_explanation, signal_length)


def extract_attention_weights(model, inputs, signal_length):
    """
    Compute a combined explanation from multiple layers.
    For each layer in EXPLANATION_LAYERS, if the layer outputs attention (or
    analogous activations), we average over the appropriate dimension and weight
    by model confidence. The final explanation is the average of all layers.
    """
    all_attention_explanations = []

    for layer_name in TARGET_LAYERS:
        layer = model.get_layer(layer_name)
        intermediate_model = tf.keras.models.Model(
            inputs=model.input, outputs=layer.output
        )
        layer_output = intermediate_model(inputs, training=False)
        # If the layer is a MultiHeadAttention, average over the head dimension.
        layer_attention = np.mean(layer_output, axis=-1)

        # Compute model confidence per example.
        # predictions = model.predict(inputs)
        # confidence = np.max(predictions, axis=-1)  # shape: (n_examples,)
        # layer_attention = layer_attention * confidence[:, None]
        # Normalize each layer's explanation.
        # layer_attention = (layer_attention - np.min(layer_attention)) / (
        #     np.max(layer_attention) - np.min(layer_attention) + 1e-8
        # )
        all_attention_explanations.append(layer_attention)

    # Average the attention explanations from all layers.
    combined_attention = np.mean(np.stack(all_attention_explanations, axis=0), axis=0)
    # mean_combined_attention = np.mean(combined_attention)
    # std_combined_attention = np.std(combined_attention)
    # z = (combined_attention - mean_combined_attention) / (std_combined_attention + 1e-8)
    # final_combined_attention = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-8)
    return expand_explanations(combined_attention, signal_length)


def plot_hrv(file_path):
    file_path = os.path.join(data_dir, file_path)
    df = preprocess_data(file_path)
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(df["RR-interval [ms]"], linewidth=0.5, color="blue")
    ax.set_xlabel("Time", fontsize=25)
    ax.set_ylabel("RR-interval [ms]", fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    return fig


def plot_explanation(signal, explanation, title, cmap="jet"):
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(signal, color="blue", linewidth=1, label="RR-interval [ms]")
    x_axis = np.arange(len(signal))
    y_min, y_max = signal.min(), signal.max()
    X, Y = np.meshgrid(x_axis, np.linspace(y_min, y_max, 50))
    Z = np.tile(explanation, (50, 1))

    im = ax.pcolormesh(X, Y, Z, cmap=cmap, alpha=0.6, shading="auto")
    fig.colorbar(im, ax=ax, label="Intensity")

    ax.set_xlim(0, len(signal))
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    plt.legend()
    plt.close()
    return fig


def dtw_alignment(attn_map, grad_map):
    """
    Align the attention map with the gradient map using Dynamic Time Warping (DTW).
    Returns the aligned attention map.
    """
    # Ensure both maps are 1D arrays (if not, reshape them)
    attn_map = np.array(attn_map).reshape(-1, 1)
    grad_map = np.array(grad_map).reshape(-1, 1)

    # Compute DTW distance and path
    distance, path = fastdtw(grad_map, attn_map)
    
    # Re-align the attention map based on the DTW path
    aligned_attn_map = np.zeros_like(attn_map)
    for i, (grad_idx, attn_idx) in enumerate(path):
        aligned_attn_map[grad_idx] = attn_map[attn_idx]

    # Flatten aligned attention map to 1D for plotting
    return aligned_attn_map.flatten()


def plot_discrepancy(signal, grad_map, attn_map, threshold=0.5):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    time = np.arange(len(signal))
    
    ax1.plot(time, attn_map, label="Attention Map", color="red")
    ax1.plot(time, grad_map, label="Gradient Map", color="green")
    ax1.set_ylabel("Explanation Intensity", fontsize=12)
    ax1.legend()
    print(f"Mean Attention Map: {np.mean(attn_map)}, Mean Gradient Map: {np.mean(grad_map)}")
    corr_coef = np.corrcoef(attn_map, grad_map)[0, 1]
    print(f"Correlation Coefficient: {corr_coef}")
    discrepancy = np.abs(attn_map - grad_map)
    mask = discrepancy > threshold
    ax2.fill_between(
        time,
        0,
        1,
        where=mask,
        color="red",
        alpha=0.3,
        transform=ax2.get_xaxis_transform(),
    )
    ax2.set_ylabel("Discrepancy", fontsize=12)
    ax2.set_xlabel("Time", fontsize=12)

    plt.tight_layout()
    return fig


def process_explanations(file_path):
    # Load and preprocess data
    file_path = os.path.join(data_dir, file_path)
    df = preprocess_data(file_path)
    X_test, _ = normalize_data(df)
    signal = df["RR-interval [ms]"].values
    signal_length = len(signal)

    # Generate explanations
    grad_weights = extract_grad_weights(model, X_test, signal_length)
    attention_weights = extract_attention_weights(model, X_test, signal_length)

    aligned_attn_map = dtw_alignment(attention_weights, grad_weights)

    # Plots
    grad_fig = plot_explanation(signal, grad_weights, "Grad-CAM Explanation")
    attn_fig = plot_explanation(signal, aligned_attn_map, "Attention Explanation")
    disc_fig = plot_discrepancy(signal, grad_weights, aligned_attn_map)

    return grad_fig, attn_fig, disc_fig


def main():
    # Get list of CSV files in the HRV_anonymized_data folder
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    # Define Gradio interface
    with gr.Blocks() as demo:
        with gr.Row():
            toggle_dark = gr.Button(value="Toggle Dark")

        toggle_dark.click(
            None,
            js="""
            () => {
                document.body.classList.toggle('dark');
            }
            """,
        )

        with gr.Row():
            with gr.Column():
                file_selector = gr.Dropdown(
                    choices=csv_files, label="Select a HRV Signal of a patient:"
                )
        with gr.Row():
            plot_output = gr.Plot(label="Input RR-Interval [ms]", show_label=False)
        file_selector.change(fn=plot_hrv, inputs=file_selector, outputs=plot_output)

        with gr.Row():
            with gr.Column():
                predict_button = gr.Button("Predict", variant="primary")
                predict_output = gr.Text(label="Prediction")
                predict_button.click(
                    fn=predict_class, inputs=file_selector, outputs=predict_output
                )
            with gr.Column():
                heatmap_button = gr.Button("Show Explanation Heatmap")
        with gr.Row():
            grad_plot = gr.Plot(label="Gradient-based Explanation")
        with gr.Row():
            attn_plot = gr.Plot(label="Attention-based Explanation")
        with gr.Row():
            discrepancy_plot = gr.Plot(label="Discrepancy")
        heatmap_button.click(
            fn=process_explanations,
            inputs=file_selector,
            outputs=[grad_plot, attn_plot, discrepancy_plot],
        )

    demo.launch()


if __name__ == "__main__":
    main()
