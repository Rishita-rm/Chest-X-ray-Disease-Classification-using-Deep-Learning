# **Chest X-ray Disease Classification using Deep Learning**

## 🚀 **Project Overview**

This project aims to build a **multi-label image classification model** using deep learning to classify chest X-ray images into **four categories**:
- **COVID-19**
- **Normal**
- **Lung Opacity**
- **Viral Pneumonia**

Using a **Convolutional Neural Network (CNN)**, this model learns from the COVID-19 Radiography dataset and demonstrates real-world **image classification** techniques that are essential for healthcare AI applications.

### 📊 **Dataset**:
- **Dataset Used**: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- **Data Type**: Medical Chest X-rays
- **Classes**: COVID, Normal, Lung Opacity, Viral Pneumonia

## 🛠️ **Technologies Used**:

- **Python 3.x**
- **TensorFlow 2.x** / **Keras**
- **NumPy** / **Pandas**
- **OpenCV** for image processing
- **Matplotlib** / **Seaborn** for data visualization
- **Kaggle API** for dataset download

## 📈 **Project Flow**

1. **Dataset Loading**: Downloading and extracting chest X-ray images from Kaggle.
2. **Exploratory Data Analysis (EDA)**: Initial analysis and visualization of dataset distribution.
3. **Data Preprocessing**:
   - Image resizing and normalization using `ImageDataGenerator`.
   - Data augmentation for better model generalization.
4. **Model Building**: Implementing a custom CNN architecture from scratch.
5. **Training**: Using batch generators and training the model on the dataset.
6. **Evaluation**: Monitoring model performance using accuracy, loss, and validation metrics.
7. **Transfer Learning (optional)**: Using pre-trained models like **ResNet50** for enhanced accuracy.

## ⚙️ **Setup & Installation**

### Prerequisites:

- Python 3.x
- TensorFlow 2.x
- Kaggle API (for dataset access)
- Jupyter or Google Colab for running the notebook

### Steps to Run the Project:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/chest-xray-disease-classification.git
   cd chest-xray-disease-classification

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Download the dataset using Kaggle API**:

- Place your Kaggle API credentials (kaggle.json) in the same directory and run the following in the terminal:

  ```bash
  !kaggle datasets download -d tawsifurrahman/covid19-radiography-database
  !unzip covid19-radiography-database.zip

4. **Run the Jupyter Notebook**:

- Open the notebook and start running the cells:
  ```bash
  jupyter notebook

5. **Training the Model**:

- Once the dataset is ready and the environment is set, you can start training the model by running the training cells in the notebook

  ## 📊 **Project Results**

The model achieved an accuracy of **83%** after 2 epochs, and is expected to reach **85-90%** by the end of training.

### **Accuracy vs Epochs (Training Progress)**

```python
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.title('Accuracy Curves')
plt.show()

### **Accuracy vs Epochs (Training Progress)**
