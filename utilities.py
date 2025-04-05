from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

class Utilities:
    """A collection of utility functions for ML preprocessing and evaluation.""" 
    @staticmethod
    def accuracy(labels, predicted):
        """Calculate accuracy percentage"""
        correct = 0
        for i in range(len(labels)):
            if labels[i] == predicted[i]:
                correct += 1
        return correct / float(len(labels)) * 100.0

    @staticmethod
    def dataset_minmax(dataset):
        """Find the min and max values for each column"""
        minmax = list()
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax
     
    @staticmethod
    def normalization(dataset, minmax):
        """Rescale dataset columns to the range 0-1"""
        for row in dataset:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    
    @staticmethod
    def minmax_scalar(X_train, X_test):
        """Normalize train and test datasets using Min-Max scaling."""
        minmax = Utilities.dataset_minmax(X_train) 
        Utilities.normalization(X_train, minmax)
        Utilities.normalization(X_test, minmax)

    @staticmethod
    def stratified_shuffle_split(X, y, n_splits=1, test_size=0.2, random_state=42):
        # Define stratified splitter
        splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

        # Split data
        for train_index, test_index in splitter.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
        return X_train, X_test, y_train, y_test

    def grid_search(parameters):
        """
        Return the a 2d array of all possible parameters

        Ex:
            parameters = [ [0.1, 0.01] , [2, 3] ] ## These are possible values for two hyperparameters

            FINAL RESULT:
                [ [0.1, 2], [0.1, 3], [0.01, 2], [0.01, 3] ]
        """
         
        # Create a mesh grid from the arrays
        grid = np.meshgrid(*parameters)
        
        # Stack the grids and reshape to get all combinations
        combinations = np.stack(grid, axis=-1).reshape(-1, len(parameters))
        
        return combinations
    
    def feature_selection(X, y, model, features, n_iters, n_removed, split=0.3):
        """
        Randomly remove features from a give dataset
        This process will be done 'n_iters' amount of times
        """
        scores = list()
        for _ in range(n_iters):
            # Randomly remove features
            removeFeatures = list(set([features[np.random.randint(0,len(features)-1)] for _ in range(n_removed)]))
            
            # Create the Train and test splits
            X_train, X_test, y_train, y_test = Utilities.stratified_shuffle_split(X.to_numpy(), y.to_numpy(), 
            test_size=split, random_state=np.random.randint(10,100))        
            
            # Train model and test
            model.fit(X_train, y_train)
            scores.append((removeFeatures, model.predict(X_test, y_test)))

        return scores

