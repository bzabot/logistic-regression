import numpy as np


def sigmoid(y):
    """Return the sigmoid of the input value or array."""
    return 1 / (1 + np.exp(-1 * y))


def binary_crossentropy(A, y):
    """Compute binary cross-entropy loss between predicted probabilities and labels."""
    A = np.clip(A, 1e-15, 1 - 1e-15)
    loss = -np.mean(y * np.log(A) + (1 - y) * np.log(1 - A))
    return loss


def gradient_descent(X, y, learning_rate=0.01, iterations=1000, return_history=False):
    """Train logistic regression weights and bias using gradient descent."""
    weights = np.zeros(X.shape[1])
    bias = 0
    history = []

    for _ in range(iterations):
        Z = np.dot(X, weights) + bias
        A = sigmoid(Z)
        error = A - y

        dw = 1 / len(y) * np.dot(X.T, error)
        db = 1 / len(y) * np.sum(error)

        weights = weights - (learning_rate * dw)
        bias = bias - (learning_rate * db)

        if return_history:
            history.append(binary_crossentropy(A, y))

    if return_history:
        return weights, bias, history

    return weights, bias


def predict_proba(X, weights, bias):
    """Compute logistic regression probabilities from features, weights, and bias."""
    return sigmoid(np.dot(X, weights) + bias)


def predict(X, weights, bias, threshold=0.5):
    """Convert probabilities into binary labels using a threshold."""
    probabilities = predict_proba(X, weights, bias)
    return (probabilities >= threshold).astype(int)
