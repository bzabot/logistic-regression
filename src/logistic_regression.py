"""Simple Logistic Regression implemented with NumPy.

This class keeps the model state inside the object:
- weights
- bias

Usage:

    model = LogisticRegression(learning_rate=0.01, iterations=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

Add-ons:

    model = LogisticRegression(addons=[Gaar(lambda_=0.01)])
    model = LogisticRegression(addons=[CAGD(alpha=5.0, epsilon=1e-3)])
    model = LogisticRegression(addons=[WeightedErrors(gamma=2.0), Gaar(lambda_=0.01)])
    model = LogisticRegression(addons=["f1_threshold"])
    model.fit(X_train, y_train, X_val, y_val)
"""

import numpy as np


class LogisticRegressionAddon:
    """Base class for optional training changes."""

    def update_error(self, X, y, probabilities, error):
        return error

    def gradient_adjustment(self, X, y, probabilities, error):
        return np.zeros(X.shape[1])

    def update_gradient(self, X, y, probabilities, dw, db):
        return dw, db

    def select_threshold(self, model, X_val, y_val):
        return None


class WeightedErrors(LogisticRegressionAddon):
    """Scales each example's error using focal-loss style weights."""

    def __init__(self, gamma=2.0, alpha=0.5):
        self.gamma = gamma
        self.alpha = alpha

    def update_error(self, X, y, probabilities, error):
        p_t = np.where(y == 1, probabilities, 1 - probabilities)
        alpha_t = np.where(y == 1, self.alpha, 1 - self.alpha)
        weights = alpha_t * (1 - p_t) ** self.gamma
        return weights * error


class Gaar(LogisticRegressionAddon):
    """Adds the GAAR gradient penalty."""

    def __init__(self, lambda_=0.01, c_minority=10.0):
        self.lambda_ = lambda_
        self.c_minority = c_minority

    def gradient_adjustment(self, X, y, probabilities, error):
        n = len(y)
        c = np.where(y == 1, self.c_minority, 1.0)
        sigmoid_deriv = probabilities * (1 - probabilities)
        x_norms_sq = np.sum(X**2, axis=1)
        scale = 2 * c * error * sigmoid_deriv * x_norms_sq
        return self.lambda_ * np.dot(X.T, scale) / n


class CAGD(LogisticRegressionAddon):
    """Preconditions weight updates with class-asymmetric Hessian curvature."""

    def __init__(self, alpha=5.0, epsilon=1e-3, approximation="full"):
        if alpha <= 0:
            raise ValueError("CAGD alpha must be positive.")

        if epsilon <= 0:
            raise ValueError("CAGD epsilon must be positive.")

        if approximation not in ("full", "diagonal"):
            raise ValueError(f"Unknown CAGD approximation: {approximation}")

        self.alpha = alpha
        self.epsilon = epsilon
        self.approximation = approximation

    def update_gradient(self, X, y, probabilities, dw, db):
        if self.approximation == "full":
            hessian_0, hessian_1 = self._class_hessians(X, y, probabilities)
            dw = self._full_preconditioned_gradient(hessian_0, hessian_1, dw)
        else:
            dw = self._diagonal_preconditioned_gradient(X, y, probabilities, dw)

        return dw, db

    def _class_hessians(self, X, y, probabilities):
        curvature = probabilities * (1 - probabilities)
        majority_mask = y == 0
        minority_mask = y == 1

        hessian_0 = self._class_hessian(X, curvature, majority_mask)
        hessian_1 = self._class_hessian(X, curvature, minority_mask)
        return hessian_0, hessian_1

    def _class_hessian(self, X, curvature, mask):
        if not np.any(mask):
            return np.zeros((X.shape[1], X.shape[1]))

        weighted_X = X[mask] * curvature[mask, None]
        return np.dot(weighted_X.T, X[mask]) / len(X)

    def _full_preconditioned_gradient(self, hessian_0, hessian_1, gradient):
        preconditioner = self._preconditioner(hessian_0, hessian_1)
        return np.linalg.solve(preconditioner, gradient)

    def _diagonal_preconditioned_gradient(self, X, y, probabilities, gradient):
        curvature = probabilities * (1 - probabilities)
        hessian_0_diagonal = self._class_hessian_diagonal(X, y == 0, curvature)
        hessian_1_diagonal = self._class_hessian_diagonal(X, y == 1, curvature)
        preconditioner_diagonal = (
            hessian_0_diagonal + self.alpha * hessian_1_diagonal + self.epsilon
        )
        return gradient / preconditioner_diagonal

    def _class_hessian_diagonal(self, X, mask, curvature):
        if not np.any(mask):
            return np.zeros(X.shape[1])

        return np.sum(curvature[mask, None] * X[mask] ** 2, axis=0) / len(X)

    def _preconditioner(self, hessian_0, hessian_1):
        n_features = hessian_0.shape[0]
        damping = self.epsilon * np.eye(n_features)
        return hessian_0 + self.alpha * hessian_1 + damping


class F1Threshold(LogisticRegressionAddon):
    """Chooses the classification threshold with the best validation F1 score."""

    def __init__(self, thresholds=None):
        self.thresholds = thresholds
        self.threshold = 0.5
        self.score = None

    def select_threshold(self, model, X_val, y_val):
        if X_val is None or y_val is None:
            raise ValueError("F1Threshold requires X_val and y_val in fit().")

        probabilities = model.predict_proba(X_val)
        self.threshold, self.score = self.find_best_f1_threshold(
            probabilities, y_val, self.thresholds
        )
        return self.threshold, self.score

    def f1_score(self, y_true, y_pred):
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        denominator = 2 * tp + fp + fn

        if denominator == 0:
            return 0.0

        return 2 * tp / denominator

    def find_best_f1_threshold(self, probabilities, y_true, thresholds=None):
        probabilities = np.asarray(probabilities, dtype=float)
        y_true = np.asarray(y_true).astype(int)

        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 100)

        best_threshold, best_score = 0.5, -1.0

        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            score = self.f1_score(y_true, predictions)

            if score > best_score:
                best_threshold, best_score = threshold, score

        return best_threshold, best_score


ADDON_FACTORIES = {
    "weighted_errors": WeightedErrors,
    "gaar": Gaar,
    "cagd": CAGD,
    "f1_threshold": F1Threshold,
}


class LogisticRegression:
    """Binary logistic regression trained with gradient descent."""

    def __init__(
        self,
        learning_rate=0.01,
        iterations=1000,
        addons=None,
    ):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0.0
        self.threshold = 0.5
        self.threshold_score = None

        self.addons = self._build_addons(addons)

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        self.threshold = 0.5
        self.threshold_score = None

        for _ in range(self.iterations):
            scores = np.dot(X, self.weights) + self.bias
            probabilities = self._sigmoid(scores)
            error = probabilities - y

            for addon in self.addons:
                error = addon.update_error(X, y, probabilities, error)

            dw = np.dot(X.T, error) / len(y)
            db = np.mean(error)

            for addon in self.addons:
                dw += addon.gradient_adjustment(X, y, probabilities, error)

            for addon in self.addons:
                dw, db = addon.update_gradient(X, y, probabilities, dw, db)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

        self._select_threshold(X_val, y_val)

        return self

    def _build_addons(self, addons):
        if addons is None:
            return []

        return [self._create_addon(addon) for addon in addons]

    def _create_addon(self, addon):
        if isinstance(addon, str):
            try:
                return ADDON_FACTORIES[addon]()
            except KeyError as exc:
                raise ValueError(f"Unknown add-on: {addon}") from exc

        return addon

    def _select_threshold(self, X_val, y_val):
        for addon in self.addons:
            threshold = addon.select_threshold(self, X_val, y_val)
            if threshold is not None:
                self.threshold, self.threshold_score = threshold

    def predict_proba(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        scores = np.dot(X, self.weights) + self.bias
        return self._sigmoid(scores)

    def predict(self, X, threshold=None):
        if threshold is None:
            threshold = self.threshold
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def loss(self, X, y):
        y = np.asarray(y, dtype=float)
        probabilities = self.predict_proba(X)
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))

    def _check_is_fitted(self):
        if self.weights is None:
            raise ValueError("Model must be fitted before prediction.")

    def _sigmoid(self, scores):
        scores = np.clip(scores, -500, 500)
        return 1 / (1 + np.exp(-scores))
