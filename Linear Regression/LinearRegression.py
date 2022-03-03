import numpy as np

# Approximation Formula
# y_hat = w.x + b

# Update rule
# w = w - alpha (learning rate) . dw
# b = b - alpha (learning rate) . db

# dj/dw = dw = 1 / M * sum of all traning example m (2).(Xi).(y_hat - yi)
# dj/db = db = 1 / M * sum of all traning example m (2).(y_hat - yi)
# 2 is a scaling factor


class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # implement gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
