import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


LAYER_NAME = "multi_head_attention"  # Replace with your convolutional layer name
SEQUENCE_LENGTH = 50
data_dir = "HRV_anonymized_data"
explanation_dir = "explanations"

# Load the pre-trained model
model = tf.keras.models.load_model("tcam_with_mine_2.keras")


def expand_explanations(explanation_2d, signal_length, seq_length=50):
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
    expanded = (expanded - np.min(expanded)) / (
        np.max(expanded) - np.min(expanded) + 1e-8
    )
    return expanded


def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df[["Phone timestamp", "RR-interval [ms]"]] = df[
        "Phone timestamp;RR-interval [ms]"
    ].str.split(";", expand=True)
    df["RR-interval [ms]"] = pd.to_numeric(df["RR-interval [ms]"])
    df.drop(columns=["Phone timestamp;RR-interval [ms]"], inplace=True)
    return df


def normalize_data(df):
    data = df["RR-interval [ms]"].values
    scaler = StandardScaler()
    data = data.reshape(-1, 1)
    normalized_data = scaler.fit_transform(data)
    sequences = []
    for i in range(len(normalized_data) - SEQUENCE_LENGTH + 1):
        sequences.append(normalized_data[i : i + SEQUENCE_LENGTH])
    return np.array(sequences), scaler


def extract_grad_weights(model, input, layer_name, signal_length):
    layer = model.get_layer(layer_name)
    x = tf.convert_to_tensor(input, dtype=tf.float32)
    intermediate_model = tf.keras.models.Model(
        inputs=model.inputs, outputs=[layer.output, model.output]
    )

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
            pred_indices = tf.argmax(predictions, axis=-1)
            pred_indices = tf.cast(pred_indices, tf.int32)
            indices = tf.stack([tf.range(tf.shape(predictions)[0]), pred_indices], axis=1)
            losses = tf.gather_nd(predictions, indices)
            loss = tf.reduce_sum(losses)

        grads = tape.gradient(loss, layer_output)
        all_layer_outputs.append(layer_output.numpy())
        all_grads.append(grads.numpy())
        # pred_indices.append(pred_index.numpy())

    # Concatenate all collected data
    all_layer_outputs = np.concatenate(all_layer_outputs, axis=0)
    all_grads = np.concatenate(all_grads, axis=0)

    # Perform normalization
    pooled_grads = np.mean(all_grads, axis=(0, 1))
    grad_weights = np.mean(np.multiply(pooled_grads, all_layer_outputs), axis=-1)

    # Normalize the grad_weights to [0, 1]
    grad_weights = (grad_weights - np.min(grad_weights)) / (
        np.max(grad_weights) - np.min(grad_weights) + 1e-8
    )

    return expand_explanations(grad_weights, signal_length)


def extract_attention_weights(model, inputs, layer_name, signal_length):
    layer = model.get_layer(layer_name)
    attention_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
    attention_weights = attention_model.predict(inputs)
    attention_weights = np.mean(attention_weights, axis=-1)
    # Normalize the attention weights to [0, 1]
    attention_weights = (attention_weights - np.min(attention_weights)) / (
        np.max(attention_weights) - np.min(attention_weights) + 1e-8
    )
    return expand_explanations(attention_weights, signal_length)


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


def predict_class(file_path):
    file_path = os.path.join(data_dir, file_path)
    df = preprocess_data(file_path)
    X_test, _ = normalize_data(df)
    y_pred = model.predict(X_test)
    predicted_class = np.argmax(y_pred, axis=-1)[0]
    predicted_text = "Control" if predicted_class == 0 else "Treatment"
    return predicted_text


def plot_discrepancy(signal, attn_map, grad_map, threshold=0.5):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    time = np.arange(len(signal))

    ax1.plot(time, attn_map, label="Attention Map", color="red")
    ax1.plot(time, grad_map, label="Gradient Map", color="green")
    ax1.set_ylabel("Explanation Intensity", fontsize=12)
    corr_coef = np.corrcoef(attn_map, grad_map)[0, 1]
    if corr_coef > 0.7:
        print(f"High correlation between explanations: {corr_coef}")
    else:
        print(f"Low correlation between explanations: {corr_coef}")
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
    input_data = np.expand_dims(X_test, axis=-1)
    signal = df["RR-interval [ms]"].values
    signal_length = len(signal)

    # Generate explanations
    grad_weights = extract_grad_weights(model, input_data, LAYER_NAME, signal_length)
    attention_weights = extract_attention_weights(
        model, X_test, LAYER_NAME, signal_length
    )

    # Plots
    grad_fig = plot_explanation(signal, grad_weights, "Grad-CAM Explanation")
    attn_fig = plot_explanation(signal, attention_weights, "Attention Explanation")
    disc_fig = plot_discrepancy(signal, grad_weights, attention_weights)

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
