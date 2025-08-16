 """
 MNIST Handwritten Digit Classification
 This notebook implements and compares two deep learning models using the MNIST handwritten 
A Fully Connected Neural Network (Dense)
 A Convolutional Neural Network (CNN)
 We:
 Preprocess data (normalization, reshaping)
 Train & evaluate both models
 Compare their performance
 Visualize misclassifications & metrics
 """
 import tensorflow as tf
 from tensorflow.keras.datasets import mnist
 from tensorflow.keras.models import Sequential
 from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
 from tensorflow.keras.utils import to_categorical
 from sklearn.metrics import classification_report, confusion_matrix
 import numpy as np
 import matplotlib.pyplot as plt
 import seaborn as sns

 # Normalize pixel values (0–255 0–1)
 # Reshape data for CNN input
# Convert labels to one-hot encoded vectors
 (x_train, y_train), (x_test, y_test) = mnist.load_data()
 # Normalize the data
 x_train = x_train / 255.0
 x_test = x_test / 255.0
 # CNN needs a 4D input: (samples, height, width, channels)
 x_train_cnn = x_train.reshape(-1, 28, 28, 1)
 x_test_cnn = x_test.reshape(-1, 28, 28, 1)
 # One-hot encode the labels
 y_train_cat = to_categorical(y_train)
 y_test_cat = to_categorical(y_test)

"""
DENSE NEURAL NETWORK (DNN)
 Architecture:
 Flatten: convert 2D image to 1D vector
 Dense(128): hidden layer with 128 neurons (ReLU)
 Dense(10): output layer for 10 digit classes (Softmax)
"""
 from tensorflow.keras.layers import Dropout
 dense_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
 ])
 dense_model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=0.5),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
 dense_history = dense_model.fit(x_train, y_train_cat,
                                epochs=3, batch_size=70,
                                validation_split=0.25)

"""
 Convolutional Neural Network (CNN)
 Architecture:
 Conv2D(32): extract 32 feature maps using 3x3 filters
 MaxPooling2D: downsample using 2x2 pool
 Dense(64): fully connected layer
 Dense(10): output softmax layer
 CNNs are better at capturing spatial patterns than dense networks
"""
cnn_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
 ])
 cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
 cnn_history = cnn_model.fit(x_train_cnn, y_train_cat,
                            epochs=5, batch_size=64,
                            validation_split=0.27)

# Compare the accuracy of both models on the unseen test dataset.
 dense_test_acc = dense_model.evaluate(x_test, y_test_cat, verbose=0)[1]
 cnn_test_acc = cnn_model.evaluate(x_test_cnn, y_test_cat, verbose=0)[1]
 print(f"Dense Model Accuracy: {dense_test_acc:.4f}")
 print(f"CNN Model Accuracy:   {cnn_test_acc:.4f}")
