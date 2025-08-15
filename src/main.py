
# main.py
# Author: Ibrahima Ba
# Project: Iris Flower Classifier with Decision Boundaries
# Description: This script loads the Iris dataset, trains a Decision Tree Classifier
#              using petal length and petal width, evaluates accuracy, and visualizes
#              both the data and the decision boundaries.

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the Iris dataset
iris = load_iris()


# Display feature names for reference
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)

# Step 2: Select features
# We use petal length (column index 2) and petal width (column index 3)
X = iris.data[:, 2:4]  # Selecting only two features for visualization
y = iris.target        # Target labels (0 = setosa, 1 = versicolor, 2 = virginica)
target_names = iris.target_names

# Step 3: Split the dataset into training and test sets
# This helps us evaluate how well the model generalizes to unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = clf.predict(X_test)

# Step 6: Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Classifier Accuracy on Test Set: {:.2f}%".format(accuracy * 100))


# Step 7: Scatter plot of petal length vs. petal width
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=target_name)

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Iris Dataset: Petal Length vs Petal Width')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 8: Plot decision boundaries
# Create a mesh grid over the feature space
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict the class for each point in the grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)

# Overlay the actual data points
for i, target_name in enumerate(target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=target_name)

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Decision Boundaries of Iris Classifier')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


