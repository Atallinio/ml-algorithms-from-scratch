import numpy as np
import pandas as pd

import time
import math
import random
from utilities import Utilities
from sklearn.metrics import confusion_matrix

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.weights = None
        self.biases = None
        self.cm = None

    def fit(self, X, y):
        # Get the rows and features from the training dataset
        n_rows, n_features = X.shape

        y_true = np.where(y<=0, -1, 1)
        # create random weights between [-1,1]
        self.weights = (np.random.rand(n_features) * 2) - 1 
        self.biases = 0
        
        # Loop over entire dataset
        for _ in range(self.n_epochs):
            for index, row in enumerate(X):
                # Margin condition check: 
                # If the sample is correctly classified with sufficient margin (≥1)
                if y_true[index]*(np.dot(row, self.weights) - self.biases) >= 1:
                    # Case 1: Sample is outside the margin (correctly classified)
                    # Only perform L2 regularization update (no hinge loss gradient)
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    # Case 2: Sample is inside margin or misclassified
                    # Update weights with both:
                    # - L2 regularization term (2*lambda*weights)
                    # - Hinge loss gradient (-y_i*x_i)      
                    self.weights -= self.learning_rate * ((2 * self.lambda_param * self.weights) - 
                    (np.dot(row, y_true[index])))
                    self.biases -= self.learning_rate * y_true[index]


    def predict(self, X):
        """
        Predict the labels
        """
        line = np.dot(X, self.weights) - self.biases 
        return np.sign(line)

class MultiClassSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.001, n_epochs=1000, labels=[1,2,3,5,6,7]):
        # Create a Linear SVM for each label
        self.labels = labels
        self.svm = {l : LinearSVM(learning_rate, lambda_param, n_epochs) for l in self.labels}
            

    def fit(self, X, y):
        """
        Train the dataset on each SVM
        """
        uniqueLabels = list(set(y))

        for label in self.labels:
            y_ = np.where(y == label, 1, -1)
            self.svm[label].fit(X, y_)
    
    def predict(self, X, y):
        """
        Predict the test set using each svm and get the best score
        """

        svmPredictions = np.zeros((X.shape[0], len(self.labels)))
        for i, label in enumerate(self.labels):
            svmPredictions[:, i] = self.svm[label].predict(X)

        predictions = np.array([self.labels[np.argmax(svmPredictions[i])] for i in range(X.shape[0])])

        self.cm = confusion_matrix(y, predictions)
        return Utilities.accuracy(y, predictions)

# Load the dataset
dataDir = "./glass+identification/glass.data"
column_names = ["ID", "RI", "NA2O", "MGO", "AL2O3", "SIO2", "K2O", "CAO", "BAO", "FE2O3", "TYPE"]
glassDataset = pd.read_csv(dataDir, index_col=0, names=column_names)

# Prepare the features and labels
X = glassDataset.drop(columns=["TYPE"])
y = glassDataset["TYPE"]

print("The MultiClass Linear SVM implementation")
print("Train/Test split: 60 / 40")

# Split the dataset using a random split based on my Student ID number
studentId= 39551288
X_train, X_test, y_train, y_test = Utilities.stratified_shuffle_split(X.to_numpy(), y.to_numpy(), 
test_size=0.40, random_state=studentId)

print("Without using normalization")
# Initialize the model and fit and predict to get accuracy
svm = MultiClassSVM()

start_time = time.time()
svm.fit(X_train, y_train)
print(f"Time to train: {time.time() - start_time}")

start_time = time.time()
accuracy = svm.predict(X_test, y_test)
print(f"Time to test: {time.time() - start_time}")
print("Accuracy: ", accuracy, "%")
print("Confusion Matrix:")
print(svm.cm)

# Normalizing the dataset using a minmax scalar
Utilities.minmax_scalar(X_train, X_test)
print("Using MinMax scalar to normalize the dataset")

# Initialize the model and fit and predict to get accuracy
svm = MultiClassSVM()
svm.fit(X_train, y_train)
accuracy = svm.predict(X_test, y_test)
print("Accuracy: ", accuracy, "%")


# Function for finding the best hyperparameters
def grid_search():
    parameters = [ [0.1, 0.01, 0.001, 0.002, 0.003], [0.1, 0.01, 0.001, 0.002, 0.003] ] 

    # Unpack values returned from the grid search function to get all possible parameter combinations
    hyper_par = Utilities.grid_search(parameters) 
    best_values = list()

    # run the model for each hyperparameter
    for h in hyper_par:
        svm = MultiClassSVM(learning_rate=h[0], lambda_param=h[1]) 
        svm.fit(X_train, y_train)
        best_values.append((svm.predict(X_test, y_test), list(h)))
    
    best_score, param = max(best_values, key= lambda x : x[0])
    print(f"Best Score: {best_score}% Using the Learning Rate {param[0]} and Lambda Parameter {param[1]}")
    return param

print("Running Grid Search to find best learning rate and lambda parameter")
param = grid_search()

print("Run the feature selection experiment using the best grid search parameter")
print("Will remove 3 ramdom features 10 times")
features = [ "RI", "NA2O", "MGO", "AL2O3", "SIO2", "K2O", "CAO", "BAO", "FE2O3"]
scores = Utilities.feature_selection(X, y, MultiClassSVM(learning_rate=param[0], lambda_param=param[1]), features=features, n_iters=10, n_removed=3, split=0.4)

# Get the best score and the features removed using that score
removed_features, best_score = max(scores, key= lambda x : x[1])
print(f"Best Score: {best_score}% , Removed Features: {removed_features}")
