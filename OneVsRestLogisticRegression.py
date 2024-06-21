from LogisticRegression import LogisticRegression
import numpy as np


class OneVsRestLogisticRegression:
    def __init__(self, learning_rate=0.0001, n_iterations=1000, thresh=0.00001):
        """
        Initializes the OneVsRestLogisticRegression model with specified hyperparameters.

        Parameters:
        learning_rate (float): Learning rate for gradient descent. Default is 0.01.
        n_iterations (int): Maximum number of iterations for gradient descent. Default is 1000.
        thresh (float): Threshold for the gradient norm to stop the gradient descent. Default is 0.001.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.thresh = thresh
        self.classifiers = []
        self.classes = None

    def fit(self, X, y):
        """
        Trains multiple logistic regression classifiers using one-vs-rest approach.

        Parameters:
        X (np.ndarray): Training feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Training labels vector of shape (n_samples,).
        """
        self.classes = np.unique(y)
        self.classifiers = []

        for c in self.classes:
            y_binary = (y == c).astype(int)  # Convert boolean array to integer array (0 and 1)
            classifier = LogisticRegression(
                learning_rate=self.learning_rate,
                n_iterations=self.n_iterations,
                thresh=self.thresh
            )
            classifier.fit(X, y_binary)
            self.classifiers.append(classifier)

    def predict(self, X):
        """
        Predicts the class labels for the input feature matrix.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
        np.ndarray: Predicted class labels vector of shape (n_samples,).
        """
        probabilities = self.predict_proba(X)
        return self.classes[np.argmax(probabilities, axis=1)]

    def predict_proba(self, X):
        """
        Predicts the class probabilities for the input feature matrix.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
        np.ndarray: Predicted class probabilities matrix of shape (n_samples, n_classes).
        """
        probabilities = np.array([classifier.predict_proba(X) for classifier in self.classifiers]).T
        return probabilities

    def score(self, X, y_true):
        """
        Computes the accuracy of the model on the provided data and labels.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y_true (np.ndarray): True labels vector of shape (n_samples,).

        Returns:
        float: Accuracy score.
        """
        y_pred = self.predict(X)
        score = np.sum(y_true == y_pred) / len(y_true)
        return score


