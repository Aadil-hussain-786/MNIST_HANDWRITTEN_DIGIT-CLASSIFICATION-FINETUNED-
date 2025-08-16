# 🤙 Handwritten Digit Recognition using Deep Learning

_A deep learning project comparing dense and convolutional networks for handwritten digit recognition using TensorFlow and the MNIST dataset._
- _DNN_ _&_ _CNN_
---

## 📂 Dataset and Tools

- **Dataset**: MNIST (70,000 images of handwritten digits, 28×28 grayscale)
- **Models Used**:
  - Dense Neural Network
  - Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Tools**: Jupyter Notebook, NumPy, Matplotlib, Seaborn

---

## 🧰 Skills Demonstrated
- Building deep learning models with TensorFlow/Keras
- Image preprocessing and reshaping for CNNs
- Performance comparison using accuracy and confusion matrix
- Visualizing misclassifications and prediction confidence

---

## 🚀 Model Architectures

### 🔷 1. Fully Connected Neural Network

| Layer   | Type    | Units | Activation |
|---------|---------|-------|------------|
| Input   | Flatten | -     | -          |
| Hidden  | Dense   | 128   | ReLU       |
| Output  | Dense   | 10    | Softmax    |

- **Optimizer**: Adam  
- **Loss**: Categorical Crossentropy  
- **Epochs**: 3 
- **batch_size**=70,
- **validation_split**=0.25
- **Accuracy Achieved**: ~96.94%

---

### 🔶 2. Convolutional Neural Network (CNN)

| Layer       | Type        | Filters | Kernel Size | Activation |
|-------------|-------------|---------|-------------|------------|
| Conv2D      | Convolution | 32      | 3×3         | ReLU       |
| MaxPooling2D| Pooling     | 2×2     |             |            |
| Flatten     |             |         |             |            |
| Dense       | Hidden      | 64      |             | ReLU       |
| Dense       | Output      | 10      |             | Softmax    |

- **Optimizer**: Adam  
- **Loss**: Categorical Crossentropy  
- **Epochs**: 5  
- **batch_size**=64,
- **validation_split**=0.27
- **Accuracy Achieved**: ~98.22%

---

## 📊 Performance Comparison

| Model | Test Accuracy |
|-------|---------------|
| Dense | 96.94%         |
| CNN   | 98.22%         |

---

## 📌 Why These Parameters?

- **ReLU**: Non-linear activation; avoids vanishing gradients.
- **Softmax**: Ideal for multi-class classification.
- **Adam**: Adaptive optimizer that combines RMSProp and momentum.
- **3×3 kernels**: Standard for capturing spatial features.
- **128 / 64 units**: Balance between performance and overfitting.

---

## 🔍 Results Visualization

- Misclassified digit samples shown using matplotlib
- Used confusion matrix (with `seaborn`) for deeper evaluation

---

## 📁 Project Structure

```
mnist-digit-recognition-cnn-tensorflow/
├── MNIST-MODEL.ipynb      # Jupyter notebook with both models
├── requirements.txt     
└── README.md                     # This file
```
---

## 🚀 Future Improvements
- Add dropout and batch normalization to improve generalization
- Experiment with different optimizers and learning rates
- Deploy the model using Streamlit or Flask as a web app
- Try transfer learning with pretrained models

 ---

