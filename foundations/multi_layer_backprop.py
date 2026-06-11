import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]], b1: List[float],
        W2: List[List[float]], b2: List[float],
        y_true: List[float]
    ) -> dict:

        # Convert to numpy arrays
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # ----------------
        # Forward pass
        # ----------------
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)

        z2 = W2 @ a1 + b2
        y_hat = z2

        loss = np.mean((y_hat - y_true) ** 2)

        # ----------------
        # Backward pass
        # ----------------

        n = len(y_true)

        # dL/dz2
        dz2 = (2 / n) * (y_hat - y_true)

        # gradients for W2 and b2
        dW2 = dz2[:, None] @ a1[None, :]
        db2 = dz2

        # gradient flowing into hidden layer
        da1 = W2.T @ dz2

        # ReLU derivative
        dz1 = da1 * (z1 > 0)

        # gradients for W1 and b1
        dW1 = dz1[:, None] @ x[None, :]
        db1 = dz1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }