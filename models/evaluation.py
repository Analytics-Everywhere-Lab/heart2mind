import tensorflow as tf
import matplotlib.pyplot as plt
from keras_tuner import RandomSearch
import pickle
from model import AdvancedHyperModel

print("################################ Evaluation: 6.1 #################################")

# Load the best model
best_model = tf.keras.models.load_model('best_model_separate_2.h5')

# Load the processed data
with open('data.pkl', 'rb') as f:
    X, y, patient_ids = pickle.load(f)

# Load the tuner results to extract hyperparameters
tuner = RandomSearch(
    hypermodel=AdvancedHyperModel(),
    objective='val_accuracy',
    max_trials=2000,
    executions_per_trial=1,
    directory='tuner_results',
    project_name='hrv_tuning',
    max_consecutive_failed_trials=2000  # Ignore mismatch shape
)

# Get the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters()[0]
print("Best Hyperparameters:")
for key, value in best_hyperparameters.items():
    print(f'{key}: {value}')

# Function to plot training history
def plot_training_history(history):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot training & validation loss values
    axs[0].plot(history['loss'])
    axs[0].plot(history['val_loss'])
    axs[0].set_title('Model Loss')
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation accuracy values
    axs[1].plot(history['accuracy'])
    axs[1].plot(history['val_accuracy'])
    axs[1].set_title('Model Accuracy')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()
    # Save pdf
    plt.savefig('training_history.pdf')

# Re-train the best model to get the training history
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Fit the model and save the history
history = best_model.fit(X, y, epochs=100, validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=1)
history_dict = history.history

with open('training_history.pkl', 'wb') as f:
    pickle.dump(history_dict, f)

# Plot the training history
plot_training_history(history_dict)
