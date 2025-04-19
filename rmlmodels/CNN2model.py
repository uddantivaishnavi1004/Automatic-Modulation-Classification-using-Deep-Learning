import os
import numpy as np
from tensorflow.keras.utils import to_categorical
import pickle
from sklearn.model_selection import train_test_split
from rmlmodels.CNN2model import Improved_CNN2Model

# Load Dataset
dataset_path = '2016.04C.multisnr.pkl'
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file {dataset_path} not found!")

with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f, encoding='latin1')

# Extract data
X, y = [], []
for key, data in dataset.items():
    if isinstance(key, tuple):
        modulation, snr = key
        X.append(data)
        y.extend([modulation] * len(data))

X = np.vstack(X)  # Shape: (samples, 2, 128)
y = np.array(y)

# Reshape input for CNN2Model (Adding channel dimension)
X_reshaped = X.reshape(X.shape[0], 2, 128, 1)  # Shape: (samples, 2, 128, 1)

# Encode labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_encoded, test_size=0.2, random_state=42)

# Initialize the CNN2Model
model = Improved_CNN2Model()


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('weights/CNN2Model.h5')
print("CNN2Model trained and saved successfully.") 