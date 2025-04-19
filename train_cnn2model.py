# import os
# import numpy as np
# import pickle
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from rmlmodels.CNN2model import Improved_CNN2Model

# # Load Dataset
# dataset_path = '2016.04C.multisnr.pkl'
# if not os.path.exists(dataset_path):
#     raise FileNotFoundError(f"Dataset file {dataset_path} not found!")

# with open(dataset_path, 'rb') as f:
#     dataset = pickle.load(f, encoding='latin1')

# # Extract data
# X, y = [], []
# snr_values = []

# for key, data in dataset.items():
#     if isinstance(key, tuple):
#         modulation, snr = key
#         X.append(data)
#         y.extend([modulation] * len(data))
#         snr_values.extend([snr] * len(data))  # Store SNR values

# X = np.vstack(X)  # Shape: (samples, 2, 128)
# y = np.array(y)
# snr_values = np.array(snr_values)

# # Reshape input for CNN2Model (Adding channel dimension)
# X_reshaped = X.reshape(X.shape[0], 2, 128, 1)  # Shape: (samples, 2, 128, 1)

# # Encode labels
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # One-hot encode labels
# y_onehot = to_categorical(y_encoded, num_classes=26)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
#     X_reshaped, y_onehot, snr_values, test_size=0.2, random_state=42
# )

# # Initialize the CNN2Model
# model = Improved_CNN2Model()

# # Compile model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# # Create weights directory if not exists
# weights_dir = 'weights'
# if not os.path.exists(weights_dir):
#     os.makedirs(weights_dir)

# # Save the model
# model.save(os.path.join(weights_dir, 'CNN2Model.h5'))
# print("CNN2Model trained and saved successfully.")
