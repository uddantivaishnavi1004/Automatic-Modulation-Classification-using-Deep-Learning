import os
import numpy as np
import tensorflow as tf
import librosa
import pywt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, LSTM, GRU, Dense, Dropout, Reshape, concatenate, 
                                     Bidirectional, Attention, BatchNormalization, LayerNormalization, MultiHeadAttention)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Data Augmentation - Adding Balanced AWGN and Fading Noise
def add_mixed_noise(data, snr_dB=-10, fade_factor=0.5):
    snr = 10 ** (snr_dB / 10)
    power = np.mean(data**2)
    noise_power = power / snr
    awgn = np.sqrt(noise_power) * np.random.randn(*data.shape)
    
    # Controlled Rayleigh fading
    fading = (fade_factor * np.random.rayleigh(size=data.shape)) + (1 - fade_factor)
    
    # Add Rician noise for low SNR
    if snr_dB < 0:
        rician_noise = np.sqrt(noise_power) * (np.random.normal(size=data.shape) + 1j * np.random.normal(size=data.shape)).real
        return (data * fading) + awgn + rician_noise
    else:
        return (data * fading) + awgn

# Compute Wavelet Transform for Robust Feature Extraction
def compute_wavelet(data, wavelet='db4', level=3):
    wavelet_features = []
    for signal in data:
        coeffs = pywt.wavedec(signal.flatten(), wavelet, level=level)
        wavelet_features.append(np.concatenate(coeffs))
    return np.array(wavelet_features)

# SNR-Weighted Loss Function
def snr_weighted_loss(y_true, y_pred, snr_values):
    snr_weights = 1 / (1 + np.exp(snr_values / 10))  # Higher weight for low SNR
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(loss * snr_weights)

# Transformer Block for Robust Sequence Learning
def transformer_block(x, num_heads=4, key_dim=128):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = LayerNormalization()(x + attn_output)
    return x

# Define the Improved Model
def Improved_LSTM_CNN(input_shape=[2, 128], stats_shape=[10], classes=26, dr=0.5):
    input_layer = Input(shape=input_shape + [1], name="input")  # (2, 128, 1)
    stats_input = Input(shape=stats_shape, name="stats_input")  # Statistical features
    
    # Multi-scale CNN Feature Extraction
    x1 = Conv2D(64, (1, 3), padding="same", activation="relu")(input_layer)
    x2 = Conv2D(64, (1, 5), padding="same", activation="relu")(input_layer)
    x3 = Conv2D(64, (1, 7), padding="same", activation="relu")(input_layer)
    x = concatenate([x1, x2, x3])  # Merge multi-scale features
    x = BatchNormalization()(x)
    x = Dropout(dr)(x)
    
    x = Conv2D(64, (2, 8), padding="valid", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dr)(x)
    x = Reshape((x.shape[1], -1))(x)  # Reshape for sequential model
    
    # Transformer Block
    x = transformer_block(x)
    
    # BiLSTM for Sequential Learning
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    
    # Attention Mechanism
    query = Dense(128)(x)
    key = Dense(128)(x)
    value = Dense(128)(x)
    attention_output = Attention()([query, key, value])
    x = concatenate([x, attention_output])
    
    # Statistical Feature Fusion with Attention
    stats_attention = Dense(128, activation="relu")(stats_input)
    stats_attention = Dense(128, activation="sigmoid")(stats_attention)
    x = concatenate([x, stats_attention])
    
    # Fully Connected Layers
    x = Dense(256, activation="relu")(x)
    x = Dropout(dr)(x)
    x = Dense(classes, activation="softmax")(x)
    
    model = Model(inputs=[input_layer, stats_input], outputs=x)
    return model

# Load Dataset Placeholder (Replace with actual dataset loading process)
def load_data():
    num_samples = 1000
    X = np.random.randn(num_samples, 2, 128, 1)  # Dummy data
    X = add_mixed_noise(X, snr_dB=-10)  # Add balanced noise
    stats_features = np.random.randn(num_samples, 10)  # Dummy statistical features
    y = np.random.randint(0, 26, num_samples)  # Dummy labels
    y = to_categorical(y, num_classes=26)  # Label smoothing
    return train_test_split(X, stats_features, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    # Load data
    X_train, X_test, stats_train, stats_test, y_train, y_test = load_data()
    
    # Initialize and Compile Model
    model = Improved_LSTM_CNN()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9)
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss=dynamic_focal_loss(epoch=0), metrics=["accuracy"])
    
    # Train Model
    model.fit([X_train, stats_train], y_train, validation_data=([X_test, stats_test], y_test), epochs=15, batch_size=32)
    
    # Evaluate Model
    test_loss, test_acc = model.evaluate([X_test, stats_test], y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Predictions and Classification Report
    y_pred = model.predict([X_test, stats_test])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes))

