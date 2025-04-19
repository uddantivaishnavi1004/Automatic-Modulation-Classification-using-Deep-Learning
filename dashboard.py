import os
import pickle
import numpy as np
import streamlit as st
import sounddevice as sd
import librosa
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
dataset_path = '2016.04C.multisnr.pkl'
if not os.path.exists(dataset_path):
    st.error(f"Dataset file not found at: {dataset_path}")
    st.write("Please ensure the dataset file exists at the specified path.")
    st.stop()

# Load the dataset
try:
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f, encoding='latin1')
    st.success("Dataset loaded successfully.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

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

# Reshape input for CNN-LSTM model (Adding channel dimension)
X_test_reshaped = X_test.reshape(X_test.shape[0], 2, 128, 1)  # Shape: (samples, 2, 128, 1)

# Encode labels
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)

# Load trained LSTM-CNN hybrid model
model_path = 'weights/LSTM_CNN.keras'
if os.path.exists(model_path):
    model = load_model(model_path)
    st.success("Model loaded successfully.")
else:
    st.error(f"Model file not found at: {model_path}")
    st.stop()

# Function to capture audio
def capture_audio(duration=5, sample_rate=22050):
    """
    Capture audio from the microphone.
    :param duration: Recording duration in seconds.
    :param sample_rate: Sampling rate (default: 22050 Hz).
    :return: Captured audio signal as a NumPy array.
    """
    st.write("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    return audio.flatten()  # Flatten the audio to 1D array

# Function to compute STFT
def compute_stft(audio, n_fft=256, hop_length=128):
    """
    Compute the Short-Time Fourier Transform (STFT) of the audio signal.
    :param audio: Input audio signal.
    :param n_fft: FFT window size.
    :param hop_length: Hop length for STFT.
    :return: STFT matrix.
    """
    stft_matrix = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    return stft_matrix

# Function to classify signal
def classify_signal(model, stft):
    """
    Classify the input signal using the trained model.
    :param model: Trained model.
    :param stft: STFT matrix of the signal.
    :return: Predicted class and confidence score.
    """
    # Ensure the STFT matrix has the correct shape
    if stft.shape[0] != 2 or stft.shape[1] != 128:
        stft = stft[:2, :128]  # Truncate or pad to match (2, 128)
    
    # Reshape to match model input shape (1, 2, 128, 1)
    stft = stft.reshape(1, 2, 128, 1)
    
    # Make prediction
    prediction = model.predict(stft)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence

# Main function for the dashboard
def main():
    st.title("Signal Modulation Classification Dashboard")
    st.write("This dashboard allows you to explore the performance of the LSTM-CNN model for signal modulation classification.")

    # Sidebar for user input
    st.sidebar.header("User Input")
    snr_filter = st.sidebar.slider("Select SNR Range", min_value=int(min(snr_values)), max_value=int(max(snr_values)), value=(int(min(snr_values)), int(max(snr_values))))

    # Filter data based on SNR
    filtered_indices = np.where((snr_values >= snr_filter[0]) & (snr_values <= snr_filter[1]))[0]
    X_filtered = X_test_reshaped[filtered_indices]
    y_filtered = y_test_encoded[filtered_indices]

    # Make predictions
    predictions = model.predict(X_filtered)
    predicted_labels = np.argmax(predictions, axis=1)

    # Display results
    st.header("Model Performance")
    st.write(f"Number of samples: {len(X_filtered)}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    conf_matrix = confusion_matrix(y_filtered, predicted_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_filtered, predicted_labels, target_names=label_encoder.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # SNR-wise Accuracy
    st.subheader("SNR-wise Accuracy")
    snr_accuracies = {}
    for snr in np.unique(snr_values):
        indices = np.where(snr_values == snr)[0]
        snr_accuracy = np.mean(y_test_encoded[indices] == predicted_labels[indices])
        snr_accuracies[snr] = snr_accuracy
    snr_df = pd.DataFrame(list(snr_accuracies.items()), columns=["SNR", "Accuracy"])
    st.line_chart(snr_df.set_index("SNR"))

    # Sample Predictions
    st.subheader("Sample Predictions")
    sample_indices = np.random.choice(len(X_filtered), 10, replace=False)
    for idx in sample_indices:
        true_label = label_encoder.inverse_transform([y_filtered[idx]])[0]
        predicted_label = label_encoder.inverse_transform([predicted_labels[idx]])[0]
        st.write(f"True Label: {true_label}, Predicted Label: {predicted_label}")

    # Add Real-Time Signal Classification Section
    st.subheader("Real-Time Signal Classification")
    if st.button("Classify Real-Time Signal"):
        # Capture audio
        audio = capture_audio(duration=5)  # Capture 5 seconds of audio
        st.write("Audio captured successfully.")

        # Preprocess audio
        stft = compute_stft(audio)
        st.write("Audio preprocessed successfully.")

        # Classify signal
        predicted_class, confidence = classify_signal(model, stft)
        st.write(f"Predicted Modulation: {label_encoder.inverse_transform([predicted_class])[0]}")
        st.write(f"Confidence: {confidence:.2f}")

# Run the dashboard
if __name__ == "__main__":
    main()