import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Step 1: Load the MNIST dataset
print("Loading the MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

# Step 2: Display a sample image
def show_sample_image(data, index):
    image = data.iloc[index].values.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f"Sample Image at Index {index}")
    plt.show()

print("Displaying a sample image...")
show_sample_image(X, 0)

# Step 3: Normalize the data
print("Normalizing the data...")
X = X / 255.0  # Normalize pixel values to the range 0-1
y = y.astype(int)  # Convert labels to integers

# Step 4: Split the data into training and test sets
print("Splitting the data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a k-Nearest Neighbors (k-NN) classifier
print("Training the k-NN classifier...")
knn_clf = KNeighborsClassifier(n_neighbors=3)  # You can try different values of n_neighbors
knn_clf.fit(X_train, y_train)

# Step 6: Evaluate the model
print("Evaluating the model...")
y_pred = knn_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step 7: Make predictions on new samples
def predict_and_show(data, index, model):
    sample = data.iloc[index].values.reshape(1, -1)
    prediction = model.predict(sample)
    print(f"Predicted label for sample at index {index}: {prediction[0]}")
    plt.imshow(sample.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {prediction[0]}")
    plt.show()

print("Making a prediction on a new sample...")
predict_and_show(X_test, 0, knn_clf)
predict_and_show(X_test, 1, knn_clf)
predict_and_show(X_test, 2, knn_clf)
