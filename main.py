import os 
import pickle
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.fftpack import fft
from scipy.ndimage import gaussian_filter  # For post-processing
from sklearn.model_selection import train_test_split


# Load Dataset
dataset_path = '2016.04C.multisnr.pkl'
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file {dataset_path} not found!")

with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f, encoding='latin1')

print("Dataset loaded successfully.")
print("Dataset keys:", dataset.keys())

# Extract test data
X_test, y_test, snr_values = [], [], []
for key, data in dataset.items():
    if isinstance(key, tuple):
        modulation, snr = key
        X_test.append(data)
        y_test.extend([modulation] * len(data))
        snr_values.extend([snr] * len(data))

X_test = np.vstack(X_test)  # Shape: (samples, 2, 128)
y_test = np.array(y_test)
snr_values = np.array(snr_values)

print(f"Extracted X_test shape: {X_test.shape}")
print(f"Unique Modulations: {np.unique(y_test)}")
print(f"Unique SNR Values: {np.unique(snr_values)}")

# Reshape input for CNN-LSTM model (Adding channel dimension)
X_test_reshaped = X_test.reshape(X_test.shape[0], 2, 128, 1)  # Shape: (samples, 2, 128, 1)

# Encode labels
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)

# Load trained LSTM-CNN hybrid model
model_path = 'weights/LSTM_CNN.keras'
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model loaded successfully.")
else:
    raise FileNotFoundError(f"Model file {model_path} not found!")

# SNR-Specific Training Function
def train_low_snr_model(model, X_train, y_train, snr_values, snr_range=(-20, 0)):
    """
    Train the model specifically on low-SNR data.
    """
    low_snr_indices = np.where((snr_values >= snr_range[0]) & (snr_values <= snr_range[1]))[0]
    X_low_snr = X_train[low_snr_indices]
    y_low_snr = y_train[low_snr_indices]

    print(f"Training on {len(X_low_snr)} low-SNR samples (SNR range: {snr_range[0]} to {snr_range[1]} dB).")
    model.fit(X_low_snr, y_low_snr, epochs=15, batch_size=32)

# Post-Processing: Smoothing Predictions
def smooth_predictions(predictions, sigma=1):
    """
    Apply Gaussian smoothing to predictions.
    """
    return gaussian_filter(predictions, sigma=sigma)

# Split data into training and testing sets
X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
    X_test_reshaped, y_test_encoded, snr_values, test_size=0.2, random_state=42
)

# Train the model on low-SNR data
train_low_snr_model(model, X_train, y_train, snr_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Apply post-processing (smoothing) to predictions
smoothed_predictions = smooth_predictions(predictions, sigma=1)
predicted_labels = np.argmax(smoothed_predictions, axis=1)

# Compute overall accuracy
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Model Accuracy: {accuracy:.4f}")

# Compute SNR-wise accuracy
snr_accuracies = {}
for snr in np.unique(snr_test):
    indices = np.where(snr_test == snr)[0]
    snr_accuracy = accuracy_score(y_test[indices], predicted_labels[indices])
    snr_accuracies[snr] = snr_accuracy

# Print SNR-wise accuracy
print("\nSNR-wise Accuracy:")
for snr, acc in sorted(snr_accuracies.items()):
    print(f"SNR {snr}: {acc:.4f}")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, predicted_labels, target_names=label_encoder.classes_))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predicted_labels)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Plot SNR-wise Accuracy
plt.figure(figsize=(10, 5))
plt.plot(sorted(snr_accuracies.keys()), [snr_accuracies[snr] for snr in sorted(snr_accuracies.keys())], marker='o', linestyle='-')
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy")
plt.title("SNR-wise Accuracy")
plt.grid(True)
plt.show()

# Sample Predictions
for i in range(10):
    print(f"Sample Prediction {i+1}: {label_encoder.inverse_transform([predicted_labels[i]])[0]}")

