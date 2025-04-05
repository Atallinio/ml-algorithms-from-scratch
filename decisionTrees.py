import numpy as np
import pandas as pd

import time
import math
import random
from utilities import Utilities
from sklearn.metrics import confusion_matrix

class DecisionTree:
    def __init__(self, max_depth=100, min_samples_in_node=1, min_impurity=0.001, num_buckets=2, n_features=None):

        self.max_depth = max_depth                      # maximum depth of the tree
        self.min_samples_in_node = min_samples_in_node  # minimum number of samples allowed to split a node
        self.min_impurity = min_impurity                # minimum impurity decrease allowed to split a node
        self.n_features = n_features                    # To add randomness
        self.num_buckets = num_buckets                  # number of buckets to use when binning the data to find the threshold
        self.node = None                                # The decision tree structure
        self.cm = None

    @staticmethod
    def entropy(labels):
        """
        Calculate the entropy of a set of labels.
        """
        classes = list(set(labels))
        P = dict()
        length = len(labels)

        for value in classes:
            P[value] = 0

        for label in labels:
            P[label] += 1

        for key, value in P.items():
            P[key] = value / length

        entropy = -sum([(p * np.log2(p)) for p in P.values() if p > 0])  # Avoid log(0)
        return entropy

    def bucketize_feature(self, feature_values, num_buckets):
        """
        Bucketize a single feature into discrete bins.

        EX: 
            feature_values = [20,22,25, 30,32,37 ,45,47]
            num_buckets = 3

            It will the range of the list --> 47 - 20 = 27
            Then divide by num_buckets --> 27 / 3 = 9
            Then for every "9" integers the value will be set to a new bucket

            THE FINAL RESULT WILL BE:
                [20,22,25, 30,32,37 ,45,48] ---> [0,0,0, 1,1,1 ,2,2]
        """
        min_val = np.min(feature_values)
        max_val = np.max(feature_values)
        bin_width = (max_val - min_val) / num_buckets

        # Create bins
        bins = [min_val + i * bin_width for i in range(num_buckets + 1)]

        # Bucketize the feature
        bucketized_features = np.digitize(feature_values, bins[:-1])  

        return bucketized_features, min_val, bin_width

    def id3(self, feature, labels, num_buckets):
        """
        Calculate the information gain for a specific feature using the ID3 algorithm.
        """
        entropy = self.entropy(labels)

        # Change the numerical data of the feature into X buckets/classes
        bucketized_feature, min_val, bin_width = self.bucketize_feature(feature, num_buckets)

        # Stores the information gain for each bucket to find the best gain
        ig_buckets = list()
        
        # For each of the different buckets/classes calculate the information gain to find the best threshold 
        for bucket in np.unique(bucketized_feature):

            # find the indices equal to the current bucket and the other indices
            bucket_indices = np.where(bucketized_feature == bucket)[0]
            other_indices = np.where(bucketized_feature != bucket)[0]
            
            # Get the bucket labels and the other labels
            bucket_labels = labels[bucket_indices]
            other_labels = labels[other_indices]
        
            bucket_fraction = (len(bucket_labels) / len(labels))

            # calculate the weighted entropy for that bucket
            weighted_entropy = bucket_fraction * self.entropy(bucket_labels) + (1 - bucket_fraction) * self.entropy(other_labels)
            information_gain = entropy - weighted_entropy

            ig_buckets.append(information_gain)
        
        # Find the largest information gain bucket
        bucket, information_gain = np.argmax(ig_buckets), np.max(ig_buckets)
        
        # calculate the threshold of that bucket
        threshold = min_val + (bucket * bin_width)

        return information_gain, threshold

    def test(self, x, labels, num_buckets):
        #    for i in range(x.shape[1]):
        #        print(np.max(x[:,i]))
        #        print(self.id3(x[:,i], labels, num_buckets))
    
        information_gains = [(self.id3(x[:, feature_index], labels, num_buckets), feature_index) 
                            for feature_index in range(x.shape[1])]

        (best_gain, best_threshold), best_feature = max(information_gains)

        print(best_feature, best_gain)
        print(x[x[:,0]>1.513].shape, x.shape)
        
    def build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        """
        # Create the tree node
        node = {
                 "feature" : None,
                 "left_node" : {},
                 "right_node": {},
                 "threshold": None,
                 "value" : None 
               }

        num_samples, num_features = X.shape

        # Check if we have reached the max depth or if we have too few samples in the node
        if (depth >= self.max_depth) or (num_samples <= self.min_samples_in_node):
            # Return the most common label
            node["value"] = np.argmax(np.bincount(y))
            return node

        if len(np.unique(y)) == 1:  # All labels are the same --> Its a leaf node
            node["value"] = y[0]
            return node

        # Get the information gain for each feature
        information_gains = [(self.id3(X[:, feature_index], y, self.num_buckets), feature_index) 
                            for feature_index in range(num_features)]

        # Extract the best information gain, feature and threshold for that feature
        (best_gain, best_threshold), best_feature = max(information_gains)

        if best_gain <= self.min_impurity:  # impurity/information_gain to low --> return label with largest count
            node["value"] = np.argmax(np.bincount(y))
            return node

        node["feature"] = best_feature
        node["threshold"] = best_threshold
        
        # Get the subset of the labels and samples
        left_data, left_labels = X[X[:,best_feature] <= best_threshold], y[X[:,best_feature] <= best_threshold]
        right_data, right_labels = X[X[:,best_feature] > best_threshold], y[X[:,best_feature] > best_threshold]
        
        # Create the left and right nodes by recusrively calling the function
        node["left_node"] = self.build_tree(left_data, left_labels, depth + 1)
        node["right_node"] = self.build_tree(right_data, right_labels, depth + 1)
        
        return node 

    def fit(self, X_train, y_train):
        """
        Train the decision tree by building the tree.
        """
        self.node = self.build_tree(X_train, y_train)

    def predict_sample(self, test_sample, node):
        """
        Predict label for a single test sample
        """

        # Check for Leaf Node 
        if node["feature"] != None:
            
            # Get feature and threshold
            feature = node["feature"]
            threshold = node["threshold"]
            
            # Check which node it will go to
            if test_sample[feature] <= threshold:
                return self.predict_sample(test_sample, node["left_node"])

            else:
                return self.predict_sample(test_sample, node["right_node"])

        # Return leaf value
        else:
            return node["value"] 
    
    def predict(self, X_test, y_test):
        """
        Predict the labels for a dataset.
        """
        predictions = [self.predict_sample(test_sample, self.node) for test_sample in X_test]
        # print(predictions)
        self.cm = confusion_matrix(y_test, predictions)
        return Utilities.accuracy(y_test, predictions) 

# Load the dataset
dataDir = "./glass+identification/glass.data"
column_names = ["ID", "RI", "NA2O", "MGO", "AL2O3", "SIO2", "K2O", "CAO", "BAO", "FE2O3", "TYPE"]
glassDataset = pd.read_csv(dataDir, index_col=0, names=column_names)

# Prepare the features and labels
X = glassDataset.drop(columns=["TYPE"])
y = glassDataset["TYPE"]

print("The Decision Tree implementation")
print("Train/Test split: 60 / 40")
# Split the dataset using a random split based on my Student ID number
studentId = 39551288
X_train, X_test, y_train, y_test = Utilities.stratified_shuffle_split(X.to_numpy(), y.to_numpy(), 
test_size=0.4, random_state=studentId)

# Function for finding the best hyperparameters
def grid_search():
    parameters = [ [20, 40, 50, 100], [3, 4, 5, 6, 8, 15], [3, 4, 6, 8], [0.01] ]

    # Unpack values returned from the grid search function to get all possible parameter combinations
    hyper_par = Utilities.grid_search(parameters) 
    best_values = list()

    # run the model for each hyperparameter
    for h in hyper_par:
        decision_tree = DecisionTree(max_depth=int(h[0]), num_buckets=int(h[1]), min_samples_in_node=int(h[2]), min_impurity=h[3])
        decision_tree.fit(X_train, y_train)
        best_values.append((decision_tree.predict(X_test, y_test), list(h)))

    best_score, param = max(best_values, key= lambda x : x[0])
    print(f"Best Score: {best_score}% Using the Max Depth {param[0]} & Number of Buckets {param[1]} &Minimum Samples in Node {param[2]} & Minium impurity {param[3]}")
    return param


# Normalizing the dataset using a minmax scalar
print("The use of normalization has no effect on the accuracy")
Utilities.minmax_scalar(X_train, X_test)

print("Run Decision tree")
decision_tree = DecisionTree()

start_time = time.time()
decision_tree.fit(X_train, y_train)
print(f"Time to train: {time.time() - start_time}")

start_time = time.time()
accuracy = decision_tree.predict(X_test, y_test)
print(f"Time to test: {time.time() - start_time}")
print("Accuracy: ", accuracy, "%")
print("Confusion Matrix:")
print(decision_tree.cm)

print("Preform Grid search to find best parameters ")
# Calculate the time taken
param = grid_search()

print("Run the feature selection experiment using the best grid search parameter")
print("Will remove 3 ramdom features 10 times")
features = [ "RI", "NA2O", "MGO", "AL2O3", "SIO2", "K2O", "CAO", "BAO", "FE2O3"]

# Calculate the time taken
start_time = time.time()
scores = Utilities.feature_selection(X, y, DecisionTree(max_depth=int(param[0]), num_buckets=int(param[1]), min_samples_in_node=int(param[2]), min_impurity=param[3]),
                                    features=features, n_iters=10, n_removed=3, split=0.4)

end_time = time.time() - start_time
print(f"Time taken to do feature selection: {end_time} seconds")

# Get the best score and the features removed using that score
removed_features, best_score = max(scores, key= lambda x : x[1])
print(f"Best Score: {best_score}% , Removed Features: {removed_features}")
