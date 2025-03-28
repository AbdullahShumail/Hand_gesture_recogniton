🖐 Hand Gesture Recognition using CNN

📌 Overview

This project focuses on Hand Gesture Recognition using a Convolutional Neural Network (CNN) trained on a 2GB dataset of various hand gesture images. The trained model is then used for real-time gesture recognition via an external camera. The project is implemented in Python using Google Colab for training due to computational constraints.

🚀 Features
✅ Preprocessing of hand gesture images
✅ CNN model implementation for classification
✅ Training on a large dataset (2GB)
✅ Real-time gesture recognition using an external camera
✅ Model evaluation with accuracy and loss metrics
✅ Visualization of training progress

🛠 Technologies Used
Python 🐍

TensorFlow/Keras 🔥

OpenCV 🎥

NumPy & Pandas 📊

Matplotlib & Seaborn 📈

Google Colab 💻

📂 Dataset
The dataset consists of various hand gesture images, which are preprocessed before training. It includes resizing, normalization, and data augmentation to enhance model generalization.

🏗 Model Architecture
The CNN model comprises:
🔹 Convolutional Layers – Feature extraction from images
🔹 Pooling Layers – Reducing dimensionality and overfitting
🔹 Fully Connected Layers – Classification of gestures
🔹 Softmax Activation – Multi-class classification

🔬 Training & Evaluation
The model is trained using Stochastic Gradient Descent (SGD) with categorical cross-entropy loss.

Loss and accuracy are monitored during training.

Results are visualized using matplotlib.

A confusion matrix is used to analyze performance.

🎥 Real-time Gesture Recognition
Once trained, the model is deployed to recognize gestures in real time using OpenCV and an external camera.

📊 Results
The model achieves high accuracy in classifying gestures, making it suitable for applications in sign language interpretation, gaming, and human-computer interaction.
