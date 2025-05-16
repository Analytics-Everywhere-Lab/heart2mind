from fastdtw import fastdtw
import numpy as np
import tensorflow as tf
from tqdm import tqdm


TARGET_LAYERS = [
    "bidirectional",
    "multi_head_attention",
]


def extract_grad_weights(model, input, signal_length):
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
    return expand_explanations(combined_explanation, signal_length)


def extract_attn_weights(model, inputs, signal_length):
    all_attention_explanations = []

    for layer_name in TARGET_LAYERS:
        layer = model.get_layer(layer_name)
        intermediate_model = tf.keras.models.Model(
            inputs=model.input, outputs=layer.output
        )
        layer_output = intermediate_model(inputs, training=False)
        # If the layer is a MultiHeadAttention, average over the head dimension.
        layer_attention = np.mean(layer_output, axis=-1)

        all_attention_explanations.append(layer_attention)

    # Average the attention explanations from all layers.
    combined_attention = np.mean(np.stack(all_attention_explanations, axis=0), axis=0)

    return expand_explanations(combined_attention, signal_length)


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
    mean_expanded = np.mean(expanded)
    std_expanded = np.std(expanded)
    z = (expanded - mean_expanded) / (std_expanded + 1e-8)
    expanded = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-8)
    return expanded


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
    for grad_idx, attn_idx in path:
        aligned_attn_map[grad_idx] = attn_map[attn_idx]

    # Flatten aligned attention map to 1D for plotting
    return aligned_attn_map.flatten()
