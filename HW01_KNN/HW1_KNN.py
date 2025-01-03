import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.k = k
        self.p = p

        self.X_train = None
        self.y_train = None

        self.ids = (340915156, 337604821)

    """
    Calculate Minkowski distance between x and y with parameter p.
    """
    def minkowski_distance(self, x, y):
        return np.sum(np.abs(x - y) ** self.p, axis=1) ** (1 / self.p)

    """
        Predict for a single point x.
    """
    def predict_point(self, x):
        # Calculate the distances
        distances = self.minkowski_distance(self.X_train, x)

        # Take the indices of k nearest points with respect to distance and then to labels
        # Have to sort all distances because of tie break...
        # I can try argpartition tuples of (d_i, y_i) but aaaaaaagh
        kni = np.lexsort((self.y_train, distances))[:self.k]

        # Take the labels of k nearest points
        knn = self.y_train[kni]

        # Histogram of k nearest labels, take the labels with max freq
        counter = np.bincount(knn).astype('uint8')
        max_labels = np.where(counter == np.max(counter))[0]

        if len(max_labels) == 1:
            return max_labels[0]

        # Tie breaking:
        # Filter the distances by labels with max nearest neighbors
        kni_filtered = kni[np.isin(knn, max_labels)]

        # return the min label with respect to distance (kni is already ordered by distance and labels)
        return self.y_train[kni_filtered[0]]



    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """

        # Assign the X and y for which we will check knn
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """

        return np.apply_along_axis(self.predict_point, axis=1, arr=X)

        ### Example code - don't use this:
        # return np.random.randint(low=0, high=2, size=len(X), dtype=np.uint8)


def main():

    print("*" * 20)
    print("Started HW1_340915156_337604821.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == "__main__":
    main()
