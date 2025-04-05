import numpy as np
import pandas as pd
import time
import math
import random
from utilities import Utilities
from sklearn.metrics import confusion_matrix

class NaiveBayes:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.cm = None

    @staticmethod
    def gaussian_probability(x, mean, std):
        """
        Calculate the Gaussian probability
        """
        # Return really small number if mean or standard deviation is equal to 0
        if mean == 0.0 or std == 0.0:
            return 0.00001
        
        # Applying the Probability distribution on X
        exponent = math.exp(-((x - mean) ** 2 / (2 * std**2)))
        return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

    @staticmethod
    def class_probabilities(labels):
        """
        Calculting the probability of each class occuring in the dataset 
        """

        probabilities = dict()

        # Get the unique labels
        unique_labels = list(set(labels))
        length = len(labels)

        for label in unique_labels:
            probabilities[label] = 0
            for row in labels:
                if row == label:
                    probabilities[label] += 1
            probabilities[label] /= length

        return probabilities

    @staticmethod
    def means(x, y):
        """
        Calculate the Mean for every class
        """

        class_means = dict()
        unique_labels = list(set(y))
        for label in unique_labels:
            class_means[label] = x[y == label].mean(axis=0)

        return class_means

    @staticmethod
    def stdev(x, y):
        """
        Calculate the Standart Deviation for every class
        """

        class_stdev = dict()
        unique_labels = list(set(y))
        for label in unique_labels:
            class_stdev[label] = x[y == label].std(axis=0)

        return class_stdev

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, y):
        """
        Predict the labels 
        """
        predictions = list()

        # Calculate statistical properties from training data:
        # - class_means: Mean of each feature per class
        # - class_stdev: Standard deviation of each feature per class
        # - class_prob: Prior probability of each class
        class_means = self.means(self.X_train, self.y_train)
        class_stdev = self.stdev(self.X_train, self.y_train)
        class_prob = self.class_probabilities(self.y_train)
        unique_labels = list(set(self.y_train))
        
        # Loop the test dataset
        for row in X:
            class_pred = list()
            for c in unique_labels:
                feature_prob = 1

                # Calculate joint probability of all features
                for i, feature in enumerate(row):
                    # Get Gaussian probability of current feature value
                    # given the class's mean and standard deviation
                    gaussian_prob = self.gaussian_probability(feature, class_means[c][i], class_stdev[c][i])
                    feature_prob *= gaussian_prob

                # Multiply by class prior probability
                feature_prob *= class_prob[c]

                # Store class label and its probability
                class_pred.append((c, feature_prob))

            # Select class with highest probability
            prediction = max(class_pred, key=lambda x: x[1])[0]
            predictions.append(prediction)
        # Compute the confusion matrix
        cm = confusion_matrix(y, predictions)
        self.cm = cm

        return Utilities.accuracy(y, predictions)


# Load the dataset
dataDir = "./glass+identification/glass.data"
column_names = ["ID", "RI", "NA2O", "MGO", "AL2O3", "SIO2", "K2O", "CAO", "BAO", "FE2O3", "TYPE"]
glassDataset = pd.read_csv(dataDir, index_col=0, names=column_names)

# Prepare the features and labels
X = glassDataset.drop(columns=["TYPE"])
y = glassDataset["TYPE"]

print("The Naive Bayes implementation")
print("Train/Test split: 60 / 40")

# Split the dataset using a random split based on my Student ID number
studentId = 39551288
X_train, X_test, y_train, y_test = Utilities.stratified_shuffle_split(X.to_numpy(), y.to_numpy(), 
test_size=0.40, random_state=studentId)

print("The use of normalization has no effect on the accuracy")
# Normalizing the dataset using a minmax scalar 
Utilities.minmax_scalar(X_train, X_test)


# Train and evaluate the Naive Bayes model
nb = NaiveBayes()

start_time = time.time()
nb.fit(X_train, y_train)
print(f"Time to train: {time.time() - start_time}")

start_time = time.time()
accuracy = nb.predict(X_test,y_test)
print(f"Time to test: {time.time() - start_time}")
print("Accuracy: ", accuracy, "%")

cm = nb.cm
print("Confusion Matrix:")
print(cm)


print("Run the feature selection experiment") 
print("Will remove 3 ramdom features 10 times")
features = [ "RI", "NA2O", "MGO", "AL2O3", "SIO2", "K2O", "CAO", "BAO", "FE2O3"]
scores = Utilities.feature_selection(X, y, NaiveBayes(), features=features, n_iters=10, n_removed=3, split=0.4)

# Get the best score and the features removed using that score
removed_features, best_score = max(scores, key= lambda x : x[1])
print(f"Best Score: {best_score}% , Removed Features: {removed_features}")
