# Automatic-Modulation-Classification-using-Deep-Learning
**ABSTRACT**
Automatic Modulation Classification (AMC) plays a vital role in wireless communication by identifying signal modulation schemes without prior information. This project proposes a deep learning-based AMC system using Convolutional Neural Networks (CNNs) trained on the RadioML 2016.04C dataset. The model processes raw in-phase (I) and quadrature (Q) signal data across a wide range of Signal-to-Noise Ratios (SNRs) to classify 26 different modulation types. Unlike traditional methods that rely on manual feature extraction, our approach automatically learns features, improving accuracy—especially in noisy environments. The system demonstrates strong performance and robustness, making it suitable for real-time applications in cognitive radio, spectrum monitoring, and IoT networks.

In the dynamic world of wireless communication, Automatic Modulation Classification (AMC) is a
key enabler of intelligent and adaptive systems, especially in cognitive radio, spectrum monitoring, and
interference detection. This project introduces a deep learning-based AMC system that utilizes
Convolutional Neural Networks (CNNs) to classify a broad variety of analog and digital modulation
schemes automatically. The model is trained using the RadioML 2016.04C dataset on in-phase (I) and
quadrature (Q) signal components under various Signal-to-Noise Ratio (SNR) conditions. The CNN- based system successfully learns appropriate features directly from raw signal data without the need for
manual feature extraction. Comprehensive experiments show the superior performance of the model, particularly in low-SNR conditions, compared to conventional machine learning methods like SVM, Decision Trees, and KNN. The results show the effectiveness of using deep learning to significantly
improve real-time, noise-robust modulation classification, highlighting its high suitability for next- generation wireless networks and resource-limited environments.

Additionally, the system proposed is effective in overcoming real-world issues in disaster-prone and
high-interference scenarios where traditional communication infrastructure could be damaged. Its
capability to operate reliably in low-SNR situations makes it an ideal candidate for use in mission- critical applications like military communications, IoT systems, and emergency response networks. Next-generation enhancements will involve minimizing computational overhead for edge deployment
on hardware like NVIDIA Jetson, and incorporating hybrid models integrating CNNs with RNNs or
Transformers for further improved classification accuracy and responsiveness under changing wireless
conditions. 

This paper illustrates how deep learning, especially CNNs, can make and enhance the process of
modulation type identification in challenging wireless environments. Through direct learning from raw
signal data, the model minimizes the use of expert-designed features and accommodates well to real- world signal and noise variations. This renders it a viable solution for contemporary communication
systems that need rapid, accurate, and automated signal classification.
