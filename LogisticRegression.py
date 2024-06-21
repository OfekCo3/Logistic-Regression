import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.0001, n_iterations=1000, thresh=0.00001):
        """
        Initializes the Logistic Regression model with given hyperparameters.

        Parameters:
        learning_rate (float): Learning rate for gradient descent. Default is 0.01.
        n_iterations (int): Maximum number of iterations for gradient descent. Default is 1000.
        thresh (float): Threshold for the gradient norm to stop the gradient descent. Default is 0.001.
        """
        self.learning_rate = learning_rate
        self.thresh = thresh
        self.max_iterations = n_iterations
        self.weights = None

    def gradient_descent(self, X, y):
        """
        Performs gradient descent to optimize the weights.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): True labels vector of shape (n_samples,).
        """
        cur_iterations = 0
        dw = self.cross_entropy_gradient(X, y)

        while np.linalg.norm(dw) > self.thresh and cur_iterations <= self.max_iterations:
            self.weights -= self.learning_rate * dw
            dw = self.cross_entropy_gradient(X, y)
            cur_iterations += 1

    def cross_entropy_gradient(self, X, y):
        """
        Computes the gradients of the cross-entropy loss with respect to the weights.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): True labels vector of shape (n_samples,).

        Returns:
        np.ndarray: Gradient with respect to weights (dw).
        """
        a = -y*self.sigmoid(-y*(X@self.weights))
        return np.dot(a, X)

    def fit(self, X, y):
        """
        Fits the logistic regression model to the training data.

        Parameters:
        X (np.ndarray): Training feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Training labels vector of shape (n_samples,).
        """
        n_features = X.shape[1]
        y = 2 * y - 1
        self.weights = np.zeros(n_features)
        self.gradient_descent(X, y)

    def predict(self, X):
        """
        Predicts binary labels for the input feature matrix.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
        np.ndarray: Predicted binary labels vector of shape (n_samples,).
        """
        y_predicted = self.predict_proba(X)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)

    def sigmoid(self, z):
        """
        Applies the sigmoid function to the input.

        Parameters:
        z (np.ndarray or float): Input value or array.

        Returns:
        np.ndarray or float: Output after applying the sigmoid function.
        """
        z = np.clip(z, -700, 700) # prevent overflow in exp
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        """
        Computes the probability of the positive class for the input feature matrix.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
        np.ndarray: Predicted probabilities vector of shape (n_samples,).
        """
        linear_model = np.dot(X, self.weights)
        return self.sigmoid(linear_model)

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

    def get_weights(self):
        """
        Returns the learned weights of the model.

        Returns:
        np.ndarray: Weights vector of shape (n_features,).
        """
        return self.weights










