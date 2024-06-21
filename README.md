# MNIST Handwritten Digit Recognition with k-Nearest Neighbors (k-NN)

This repository contains a Python script that demonstrates the process of training and evaluating a k-Nearest Neighbors (k-NN) classifier on the MNIST dataset for handwritten digit recognition.

## Overview

The MNIST dataset is a popular dataset used in machine learning for benchmarking classification algorithms. It consists of 70,000 small square 28x28 pixel grayscale images of handwritten single digits between 0 and 9.

### Steps in the Code

1. **Load the MNIST dataset**: The script loads the MNIST dataset using `fetch_openml` from `sklearn.datasets`.

2. **Display a sample image**: It includes a function to display a sample image from the dataset to give a visual understanding of the data.

3. **Normalize the data**: Pixel values of the images are normalized to the range 0-1 to facilitate training.

4. **Split the data**: The dataset is split into training and test sets using `train_test_split` from `sklearn.model_selection`.

5. **Train the k-NN classifier**: A k-Nearest Neighbors classifier (`KNeighborsClassifier`) is trained on the training data.

6. **Evaluate the model**: The model's accuracy is evaluated on the test set using `accuracy_score` and further analyzed using `classification_report` and a `confusion_matrix`.

7. **Make predictions**: The trained model is used to make predictions on new samples from the test set, and the results are displayed along with the predicted digit images.

## Requirements

- Python 3.x
- numpy
- matplotlib
- scikit-learn
- seaborn

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/mnist-knn.git
   cd mnist-knn

![image](https://github.com/JaswanthProjects/Handwritten-Digit-Recognition/assets/85422176/9955d112-bc14-4d44-8e42-ac147dd08531)
![image](https://github.com/JaswanthProjects/Handwritten-Digit-Recognition/assets/85422176/d90cb415-0132-4887-b549-124916ca9a49)
![image](https://github.com/JaswanthProjects/Handwritten-Digit-Recognition/assets/85422176/27681a69-0711-4157-bf2a-eb643ee4b253)




