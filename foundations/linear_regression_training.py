import numpy as np
from numpy.typing import NDArray


class Solution:
    learning_rate = 0.01

    def get_derivative(
        self,
        model_prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
        N: int,
        X: NDArray[np.float64],
        desired_weight: int
    ) -> float:
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(
        self,
        X: NDArray[np.float64],
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        w = initial_weights.copy()
        N = len(X)

        for _ in range(num_iterations):
            pred = self.get_model_prediction(X, w)

            gradients = np.zeros_like(w)

            for j in range(len(w)):
                gradients[j] = self.get_derivative(pred, Y, N, X, j)

            w = w - self.learning_rate * gradients

        return np.round(w, 5)