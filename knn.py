import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import time
import math
import random
from utilities import Utilities
from sklearn.metrics import confusion_matrix

class KNN:
    def __init__(self, num_neighbours):
        self.cm = None
        self.X_train = None
        self.y_train = None
        self.num_neighbours = num_neighbours

    def euclidean_distance(self, x1, x2):
        """
        Calculates the Euclidean distance

        Formula:
            sqrt(Σ(x1_i - x2_i)²) where i ranges over all features
        """
        return math.sqrt(sum([(fX1 - fX2) ** 2 for fX1, fX2 in zip(x1, x2)]))

    def get_neighbours(self, newData):
        """
        Identifies k-nearest neighbors to a new data point.

        Process:
            1. Computes distances to all training points
            2. Sorts points by distance
            3. Selects top-k closest points
        """
        distances = list()
        for i, row in enumerate(self.X_train):
            # Calculate distance to each training point
            distance = self.euclidean_distance(row, newData)
            distances.append((i, distance))
        
        distances.sort(key=lambda x: x[1])

        # Extract indices of k-nearest neighbors
        neighbours = [distances[n][0] for n in range(self.num_neighbours)]

        return neighbours

    def predict_one_data_point(self, test):
        """
        Predict the label on test sample using majority vote
        """
        # Get neighbours and collect neighbour labels
        neighbours = self.get_neighbours(test)
        neighbour_labels = [self.y_train[i] for i in neighbours]

        # Determine most common label
        unique_values, counts = np.unique(neighbour_labels, return_counts=True)
        prediction = unique_values[np.argmax(counts)]
        return prediction

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, y):
        """
        Predict the labels
        """
        predictions = [self.predict_one_data_point(test) for test in X]
        # Compute the confusion matrix
        cm = confusion_matrix(y, predictions)
        self.cm = cm
        return Utilities.accuracy(y, predictions)
        

# Load the dataset
dataDir = "./glass+identification/glass.data"
column_names = ["ID", "RI", "NA2O", "MGO", "AL2O3", "SIO2", "K2O", "CAO", "BAO", "FE2O3", "TYPE"]
glassDataset = pd.read_csv(dataDir, index_col=0, names=column_names )

# Prepare the features and labels
X = glassDataset.drop(columns=["TYPE"])
y = glassDataset["TYPE"]

print("The KNN implementation")
print("Train/Test split: 60 / 40")

# Split the dataset using a random split based on my Student ID number
studentId = 39551288 
X_train, X_test, y_train, y_test = Utilities.stratified_shuffle_split(X.to_numpy(), y.to_numpy(), 
test_size=0.40, random_state=studentId) 

print("Without using normalization")
# Train and evaluate the KNN model using different number of neighbours and get highest score
accuracy_scores = list()
for n in range(2,6):
    knn = KNN(num_neighbours=n)

    start_time = time.time()
    knn.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    predictions = knn.predict(X_test,y_test)
    test_time = time.time() - start_time

    accuracy_scores.append((train_time, test_time, n, knn.predict(X_test,y_test)))

train_time, test_time, neighbours, best_score = max(accuracy_scores, key=lambda x : x[1])

print(f"Time to train: {train_time}")
print(f"Time to test: {test_time}")
print(f"Accuracy:  {best_score}% Using {neighbours} Neighbours")

# Normalizing the dataset using a minmax scalar
Utilities.minmax_scalar(X_train, X_test)
print("Using MinMax scalar to normalize the dataset")

# Train and evaluate the KNN model using different number of neighbours and get highest score
accuracy_scores = list()
for n in range(2,6):
    knn = KNN(num_neighbours=n)
    knn.fit(X_train, y_train)
    accuracy_scores.append((n, knn.predict(X_test,y_test), knn.cm))
    
neighbours, best_score, cm = max(accuracy_scores, key=lambda x : x[1])

print(f"Accuracy:  {best_score}% Using {neighbours} Neighbours")
print("Confusion Matrix:")
print(cm)


print(f"Run the feature selection experiment using {neighbours} Neighbours") 
print("Will remove 1 ramdom features 20 times")
features = [ "RI", "NA2O", "MGO", "AL2O3", "SIO2", "K2O", "CAO", "BAO", "FE2O3"]
scores = Utilities.feature_selection(X, y, KNN(num_neighbours=neighbours), features=features, n_iters=20, n_removed=1, split=0.4)

# Get the best score and the features removed using that score
removed_features, best_score = max(scores, key= lambda x : x[1])
print(f"Best Score in Feature removal: {best_score}% , Removed Features: {removed_features}")


