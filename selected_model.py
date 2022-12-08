import numpy as np

# Class for KNN Regressor Implementation
class knn():
    def __init__(self, num_neighbors = 5):
        self.num_neighbors = num_neighbors
        self.X_train = None
        self.Y_train = None
     
    # Storing the Data
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
    
    # calculating the distances from all records in training set to a test example
    def distance_calculator(self, X_train, X_test):
        distance = np.sqrt(np.sum(np.square(X_train - X_test), axis = 1))
        return distance
    
    # Finding the indices of k nearest neighbors
    def neighbour_finder(self, X_train, X_test, Y_train):
        distances = self.distance_calculator(X_train, X_test)
        sorted_distances = np.sort(distances)[:self.num_neighbors]
        index = []
        for sd in sorted_distances:
            index.append([l.tolist() for l in np.where(distances == sd)])
        index=sum(sum(index,[]),[])   
        labels = Y_train[index]
        return np.mean(labels)
    
    # Predicting using KNN
    def predict(self, X_test):
        op = []
        for rec in X_test:
            op.append(self.neighbour_finder(self.X_train, rec, self.Y_train))
        return op