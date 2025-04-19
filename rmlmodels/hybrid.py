import numpy as np
import tensorflow as tf
from rmlmodels.CNN2model import CNN2Model

from LSTM_CNN_Model import LSTM_CNN

class HybridModel:
    def __init__(self, cnn_weights=None, lstm_weights=None, input_shape=[2, 128], classes=26):
        """
        Initializes both CNN2Model and LSTM-CNN Hybrid models.
        """
        self.cnn_model = CNN2Model(weights=cnn_weights, input_shape=input_shape, classes=classes)
        self.lstm_model =LSTM_CNN(input_shape=input_shape, classes=classes)
        
        if lstm_weights:
            self.lstm_model.load_weights(lstm_weights)
        
    def predict(self, X, snr_values):
        """
        Runs predictions on both models and chooses the best one based on confidence and SNR.
        """
        cnn_preds = self.cnn_model.predict(X)
        lstm_preds = self.lstm_model.predict(X)
        
        final_preds = []
        for i, snr in enumerate(snr_values):
            cnn_confidence = np.max(cnn_preds[i])
            lstm_confidence = np.max(lstm_preds[i])
            
            # Use confidence-weighted decision
            if snr > 0 and cnn_confidence > lstm_confidence:
                final_preds.append(np.argmax(cnn_preds[i]))
            else:
                final_preds.append(np.argmax(lstm_preds[i]))
        
        return np.array(final_preds)

if __name__ == "__main__":
    hybrid_model = HybridModel(cnn_weights='weights/cnn2model.keras', lstm_weights='weights/lstm_cnn.keras')
    print("Hybrid model initialized successfully!")
