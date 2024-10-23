# Tomato Leaf Disease Detection

This repository provides a machine learning-based solution for detecting various diseases in tomato leaves using a Convolutional Neural Network (CNN). The objective is to accurately classify tomato leaf images into different disease categories, facilitating early diagnosis and timely intervention in agricultural settings.

## Dataset

The dataset used for this project consists of images of tomato leaves, categorized into various classes representing different types of diseases and healthy conditions. The classes include:

- **Tomato Mosaic Virus**
- **Tomato Yellow Leaf Curl Virus**
- **Tomato Bacterial Spot**
- **Tomato Target Spot**
- **Healthy**

### Dataset Structure
The dataset is organized into separate folders for each class, containing images that are used for training, validation, and testing. This folder structure is essential for automated data loading during model training.

### Data Preprocessing
1. **Resizing**: All images are resized to a fixed dimension (e.g., 150x150 pixels) to ensure consistency across the dataset.
2. **Normalization**: The pixel values of the images are scaled to a range of 0-1 by dividing each value by 255, which helps improve model convergence during training.
3. **Augmentation**: Data augmentation techniques such as rotation, width/height shift, shear, zoom, and horizontal flipping are applied to artificially expand the dataset size and enhance the model's generalization capabilities.
4. **Splitting**: The dataset is divided into training, validation, and test sets to effectively evaluate the model’s performance. Typically, 70% of the data is used for training, 20% for validation, and 10% for testing.

### Source
The dataset can be sourced from online platforms like Kaggle, where various tomato leaf disease image datasets are available.

## Overview of the Code

The code is structured to provide a step-by-step approach for building, training, and evaluating a CNN model for tomato leaf disease detection:

1. **Importing Libraries**: Necessary libraries for deep learning, data processing, and visualization (e.g., TensorFlow, Keras, NumPy, Matplotlib) are imported.
2. **Data Preprocessing**: Using `ImageDataGenerator` for data augmentation and creating generators for training and validation datasets.
3. **Model Architecture**: The model is built using a sequential CNN structure that includes convolutional layers for feature extraction, pooling layers for dimensionality reduction, and fully connected layers for classification.
4. **Model Training**: The model is trained with training and validation datasets over multiple epochs, using metrics like accuracy and loss to monitor performance.
5. **Evaluation and Visualization**: The model's accuracy and loss are evaluated on the test dataset, and training history is visualized through plots.
6. **Prediction**: A function is provided to make predictions on new images, allowing for real-time disease detection.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Jupyter Notebook (optional)


## Acknowledgments

- The dataset used in this project is sourced from Kaggle.
- TensorFlow and Keras libraries are utilized for building and training the CNN model.

