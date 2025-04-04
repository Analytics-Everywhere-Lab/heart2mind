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
import seaborn as sns
from scipy import signal
from chatbot import PDChatbot
from explanation_functions import (
    dtw_alignment,
    extract_attn_weights,
    extract_grad_weights,
)


custom_objects = {
    "SpectralNormalization": tfa.layers.SpectralNormalization,
    "GroupNormalization": tfa.layers.GroupNormalization,
    "F1Score": tfa.metrics.F1Score,
    "BinaryFocalCrossentropy": tf.keras.losses.BinaryFocalCrossentropy,
}


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


def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df[["Phone timestamp", "RR-interval [ms]"]] = df[
        "Phone timestamp;RR-interval [ms]"
    ].str.split(";", expand=True)
    df["RR-interval [ms]"] = pd.to_numeric(df["RR-interval [ms]"])
    df.drop(columns=["Phone timestamp;RR-interval [ms]"], inplace=True)
    return df


def plot_hrv(file_path):
    sns.set_theme(style="whitegrid")
    file_path = os.path.join(data_dir, file_path)
    df = preprocess_data(file_path)
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(df["RR-interval [ms]"], color="blue", lw=1)
    ax.set_xlabel("Time", fontsize=16)
    ax.set_ylabel("RR-interval [ms]", fontsize=16)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
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


def get_contiguous_regions(mask, gap=50):
    """
    Given a boolean mask and an integer gap,
    return a list of (start_index, end_index) for each region of True values.
    Two True segments are merged into one region if they are within 'gap' indices of each other.
    """
    in_region = False
    starts, ends = [], []

    for i, val in enumerate(mask):
        if val and not in_region:
            # Entering a region
            starts.append(i)
            in_region = True
        elif not val and in_region:
            # Leaving a region
            ends.append(i - 1)
            in_region = False

    # If mask ended while still in a region, close it out
    if in_region:
        ends.append(len(mask) - 1)

    # Now 'starts' and 'ends' are contiguous raw segments.
    # We'll merge any segments that are within 'gap' of each other.
    merged_regions = []
    if not starts:
        return merged_regions  # empty list

    current_start = starts[0]
    current_end = ends[0]

    for idx in range(1, len(starts)):
        next_start = starts[idx]
        next_end = ends[idx]
        # Check if next region is within 'gap' of current region
        if (next_start - current_end - 1) <= gap:
            # Merge
            current_end = next_end
        else:
            # No merge; push the current region and start a new one
            merged_regions.append((current_start, current_end))
            current_start = next_start
            current_end = next_end

    # Add the final region
    merged_regions.append((current_start, current_end))

    return merged_regions


def plot_figures(signal, grad_map, attn_map, threshold=0.5):
    """
    Creates and returns three separate figures:
      1) Grad-CAM explanation + Attention explanation
      2) Grad vs. Attention plots + Discrepancy map
      3) Original RR signal with discrepancy overlay
    """
    discrepancy = np.abs(attn_map - grad_map)
    mask = discrepancy > threshold
    merged_regions = get_contiguous_regions(mask, gap=50)

    ### --- PART 1: Grad-CAM & Attention Explanation ---
    sns.set_theme(style="whitegrid")
    fig1 = plt.figure(figsize=(20, 8))

    time = np.arange(len(signal))
    y_min, y_max = signal.min(), signal.max()

    # Grad-CAM Explanation (top)
    ax1 = fig1.add_subplot(2, 1, 1)
    ax1.plot(time, signal, color="blue", linewidth=1, label="RR-interval [ms]")
    X, Y = np.meshgrid(time, np.linspace(y_min, y_max, 50))
    Z_grad = np.tile(grad_map, (50, 1))
    im1 = ax1.pcolormesh(X, Y, Z_grad, cmap="jet", alpha=0.6, shading="auto")
    ax1.set_xlim(0, len(signal))
    ax1.set_ylim(y_min, y_max)
    ax1.set_title("Gradient-based Explanation", fontsize=14)
    ax1.legend(loc="upper right")

    # Attention-based Explanation (bottom)
    ax2 = fig1.add_subplot(2, 1, 2, sharex=ax1)
    ax2.plot(time, signal, color="blue", linewidth=1, label="RR-interval [ms]")
    Z_attn = np.tile(attn_map, (50, 1))
    ax2.pcolormesh(X, Y, Z_attn, cmap="jet", alpha=0.6, shading="auto")
    ax2.set_xlim(0, len(signal))
    ax2.set_ylim(y_min, y_max)
    ax2.set_title("Attention-based Explanation", fontsize=14)
    ax2.legend(loc="upper right")

    # Add a shared colorbar
    cbar_ax = fig1.add_axes([0.125, 0.50, 0.775, 0.02])
    fig1.colorbar(im1, cax=cbar_ax, orientation="horizontal")

    fig1.tight_layout()
    fig1.subplots_adjust(hspace=0.5)

    ### --- PART 2: Grad vs. Attention & Discrepancy ---
    sns.set_theme(style="whitegrid")
    fig2 = plt.figure(figsize=(20, 8))

    # Grad vs. Attention
    ax3 = fig2.add_subplot(2, 1, 1)
    ax3.plot(time, grad_map, label="Gradient Map", lw=1)
    ax3.plot(time, attn_map, label="Attention Map", lw=1)
    ax3.set_ylim(0, 1)
    ax3.set_xlim(0, len(time))
    ax3.set_ylabel("Explanation Intensity [0-1]", fontsize=12)
    ax3.set_title("Gradient-based vs. Attention-based Explanations", fontsize=14)
    ax3.legend(loc="upper right")

    # Discrepancy map
    ax4 = fig2.add_subplot(2, 1, 2, sharex=ax3)
    ax4.plot(time, discrepancy, label="|Grad - Attn|", lw=1, color="green")
    ax4.fill_between(
        time,
        0,
        discrepancy,
        where=mask,
        color="red",
        alpha=0.3,
        label=f"Discrepancy > {threshold}",
    )
    ax4.set_ylim(0, 1)
    ax4.set_ylabel("Discrepancy [0-1]", fontsize=12)
    ax4.set_title("Discrepancy Mask Regions", fontsize=14)
    ax4.legend(loc="upper right")

    fig2.tight_layout()

    ### --- PART 3: RR Signal with Discrepancy Overlay ---
    sns.set_theme(style="whitegrid")
    fig3 = plt.figure(figsize=(20, 4))

    ax5 = fig3.add_subplot(1, 1, 1)
    ax5.plot(time, signal, label="RR-interval [ms]", lw=1, color="blue")

    for region_id, (start, end) in enumerate(merged_regions, 1):
        ax5.axvspan(start, end, color="red", alpha=0.3)
        region_center = (start + end) / 2
        y_pos = signal.min() + (signal.max() - signal.min()) * 0.9
        ax5.text(
            region_center,
            y_pos,
            f"{region_id}",
            horizontalalignment="center",
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
        )

    ax5.set_xlabel("Time Index", fontsize=12)
    ax5.set_ylabel("RR-interval [ms]", fontsize=12)
    ax5.set_xlim(0, len(signal))
    ax5.set_title("RR-Interval with Discrepancy Regions", fontsize=14)
    ax5.legend(loc="upper right")

    fig3.tight_layout()

    # Return the three distinct figure objects
    return fig1, fig2, fig3


def compute_hrv_metrics(rr_signal):
    """
    Compute HRV metrics (mean RR, RMSSD, SDNN, pNN50, LF/HF power) for a given RR interval signal.

    Parameters:
    -----------
    rr_signal : array-like
        RR interval signal in milliseconds

    Returns:
    --------
    dict
        Dictionary containing the computed HRV metrics
    """
    # Mean RR
    mean_rr = np.mean(rr_signal)

    # RMSSD (Root Mean Square of Successive Differences)
    diffs = np.diff(rr_signal)
    rmssd = np.sqrt(np.mean(diffs**2)) if len(diffs) > 0 else 0.0

    # SDNN (Standard Deviation of NN intervals)
    sdnn = np.std(rr_signal)

    # pNN50 (percentage of successive RR intervals that differ by more than 50 ms)
    pnn50 = 0.0
    if len(diffs) > 0:
        pnn50 = 100.0 * np.sum(np.abs(diffs) > 50) / len(diffs)

    # Frequency domain: Welch's PSD
    fs = 4.0  # Sampling frequency
    nperseg = min(len(rr_signal), 256)
    if nperseg > 1:  # Ensure we have enough data points
        freqs, psd = signal.welch(rr_signal, fs=fs, nperseg=nperseg)

        # Typical HRV frequency bands (in Hz)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.40)

        lf_mask = (freqs >= lf_band[0]) & (freqs < lf_band[1])
        hf_mask = (freqs >= hf_band[0]) & (freqs < hf_band[1])

        lf_power = np.trapz(psd[lf_mask], x=freqs[lf_mask]) if np.any(lf_mask) else 0.0
        hf_power = np.trapz(psd[hf_mask], x=freqs[hf_mask]) if np.any(hf_mask) else 0.0
    else:
        lf_power = 0.0
        hf_power = 0.0

    # Return metrics as a dictionary
    metrics = {
        "mean_rr": round(mean_rr, 2),
        "RMSSD": round(rmssd, 2),
        "SDNN": round(sdnn, 2),
        "pNN50": round(pnn50, 4),
        "LF_power": round(lf_power, 2),
        "HF_power": round(hf_power, 2),
    }

    return metrics


def compute_hrv_in_discrepancy(rr_signal, grad_map, attn_map, threshold=0.5):
    """
    Compute HRV metrics for each contiguous region where discrepancy > threshold.

    Parameters:
    -----------
    rr_signal : array-like
        RR interval signal in milliseconds
    grad_map : array-like
        Gradient-based explanation weights
    attn_map : array-like
        Attention-based explanation weights
    threshold : float, optional
        Threshold for considering discrepancy significant, by default 0.5

    Returns:
    --------
    list
        List of dictionaries containing metrics for each region
    numpy.ndarray
        Boolean mask of points where discrepancy > threshold
    """
    import numpy as np

    # Calculate discrepancy between explanation methods
    discrepancy = np.abs(attn_map - grad_map)
    mask = discrepancy > threshold

    # Find contiguous segments of True in 'mask'
    regions = get_contiguous_regions(mask, gap=50)
    if not regions:
        # No region above threshold
        return [], mask  # return an empty list + mask

    # For storing results from each region
    region_metrics_list = []

    # Compute metrics for each region
    for idx, (start, end) in enumerate(regions, 1):
        # Slice the RR-intervals for this region
        rr_roi = rr_signal[start : end + 1]

        # Skip regions with insufficient data
        if len(rr_roi) <= 1:
            continue

        # Compute HRV metrics for this region
        metrics = compute_hrv_metrics(rr_roi)

        # Skip regions with zero pNN50 (likely not physiologically meaningful)
        if metrics["pNN50"] == 0.0:
            continue

        # Add region metadata
        metrics["region_id"] = idx
        metrics["start_idx"] = start
        metrics["end_idx"] = end
        metrics["pNN50"] = f"{metrics['pNN50']}%"  # Format as percentage

        region_metrics_list.append(metrics)

    return region_metrics_list, mask


def process_explanations(file_path, chat_history, model_prediction):
    """
    Process explanations, compute HRV metrics, and update chat history.
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
    chat_history : list
        Chat history to be updated
    model_prediction : str
        Prediction from the model
        
    Returns:
    --------
    tuple
        (fig1, fig2, fig3, updated_chat_history)
    """ 
    # Load and preprocess data
    file_path = os.path.join(data_dir, file_path)
    df = preprocess_data(file_path)
    X_test, _ = normalize_data(df)
    signal = df["RR-interval [ms]"].values
    signal_length = len(signal)
    
    # Generate explanations
    grad_weights = extract_grad_weights(model, X_test, signal_length)
    attn_weights = extract_attn_weights(model, X_test, signal_length)
    aligned_attn_map = dtw_alignment(attn_weights, grad_weights)
    
    # Create figures for visualization
    fig1, fig2, fig3 = plot_figures(
        signal, grad_weights, aligned_attn_map, threshold=0.5
    )
    
    # Compute HRV metrics for the whole signal
    whole_signal_metrics = compute_hrv_metrics(signal)
    
    # Compute HRV metrics for discrepancy regions
    region_metrics_list, mask = compute_hrv_in_discrepancy(
        signal, grad_weights, aligned_attn_map, threshold=0.5
    )
    
    # First chat bubble - Model prediction and whole signal metrics
    model_pred_text = f"Model prediction: {model_prediction}\n\n"
    whole_signal_text = "HRV Metrics:\n"
    whole_signal_text += f"Mean RR: {whole_signal_metrics['mean_rr']} ms, "
    whole_signal_text += f"RMSSD: {whole_signal_metrics['RMSSD']}, "
    whole_signal_text += f"SDNN: {whole_signal_metrics['SDNN']}, "
    whole_signal_text += f"pNN50: {whole_signal_metrics['pNN50']}%\n"
    whole_signal_text += f"LF Power: {whole_signal_metrics['LF_power']}, "
    whole_signal_text += f"HF Power: {whole_signal_metrics['HF_power']}"
    
    # Append first bubble to chat history
    chat_history.append({"role": "assistant", "content": whole_signal_text, "metadata": {"title": model_pred_text}})
    
    # Second chat bubble - Discrepancy regions
    if region_metrics_list:
        discrepancy_text = "Discrepancy Regions Detected:\n"
        
        # Add each region's metrics
        for region in region_metrics_list:
            discrepancy_text += (
                f"• Region {region['region_id']} (indices {region['start_idx']}–{region['end_idx']}):\n"
                f"Mean RR: {region['mean_rr']} ms, RMSSD: {region['RMSSD']}, SDNN: {region['SDNN']}, pNN50: {region['pNN50']}\n"
                f"LF Power: {region['LF_power']}, HF Power: {region['HF_power']}\n"
            )
    else:
        # No high-discrepancy region found
        discrepancy_text = "No discrepancy region above threshold was found."
    
    # Append second bubble to chat history
    chat_history.append({"role": "assistant", "content": discrepancy_text, "metadata": {"title": "Discrepancy Regions"}})
    
    return fig1, fig2, fig3, chat_history


def main():
    # Get list of CSV files in the HRV_anonymized_data folder
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    chatbot = PDChatbot(
        "meta/llama-3.3-70b-instruct",
        "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",
    )
    # Define Gradio interface
    with gr.Blocks(theme=gr.themes.Base()) as demo:
        gr.HTML(
            "<h1 class='title'><center>Psychiatric Disorder Detection using HRV Signals</center></h1>"
        )
        gr.HTML(
            "<center><p class='subtitle'>This application uses a deep learning model to predict psychiatric disorder from HRV signals. "
            "You can select a HRV signal from the dropdown list, visualize the signal, predict the class, and view explanations "
            "and discrepancies between the model's attention and gradient-based explanations.</p></center>"
        )
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    file_selector = gr.Dropdown(
                        choices=csv_files, label="Select a HRV signal data:"
                    )
            with gr.Row():
                plot_output = gr.Plot(label="Input RR-Interval [ms]", show_label=False)
            file_selector.change(fn=plot_hrv, inputs=file_selector, outputs=plot_output)

        with gr.Group():
            with gr.Row(equal_height=True):
                with gr.Column():
                    predict_button = gr.Button("Predict", variant="primary")
                    heatmap_button = gr.Button("Show Explanations & Discrepancy")
                with gr.Column():
                    predict_output = gr.Text(label="Prediction")
            with gr.Row():
                part1_plot = gr.Plot(
                    label="[Heatmap] Gradient-based and Attention-based Explanations"
                )
            with gr.Row():
                part2_plot = gr.Plot(
                    label="[Discrepancy] Gradient-based and Attention-based Explanations"
                )
            with gr.Row():
                part3_plot = gr.Plot(label="[Diagnosis] RRI with Discrepancy Regions")

        with gr.Group():
            with gr.Row():
                chat_interface = gr.Chatbot(
                    type="messages",
                    value=[
                        {"role": "assistant", "content": "Hello! I am your AI assistant to discuss results of psychiatric disorder detection."},
                    ],
                    show_copy_button=True,
                    show_share_button=True,
                    resizable=True,
                    avatar_images=[None, "assets/bot.png"],
                    editable="user",
                    render_markdown=True,
                )
            with gr.Row(equal_height=True):
                msg = gr.Textbox(
                    placeholder="Type your message here",
                    show_label=False,
                    submit_btn=True,
                )

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
        predict_button.click(
            fn=predict_class, inputs=file_selector, outputs=predict_output
        )
        heatmap_button.click(
            fn=process_explanations,
            inputs=[file_selector, chat_interface, predict_output],
            outputs=[part1_plot, part2_plot, part3_plot, chat_interface],
        )

        msg.submit(
            chatbot.generate_response,
            inputs=[chat_interface, msg],
            outputs=chat_interface,
            show_progress="hidden",
        ).then(lambda: "", None, msg)

    demo.queue().launch(debug=True)


if __name__ == "__main__":
    main()
