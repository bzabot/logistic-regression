"""Metrics for imbalanced binary classification.

Project convention:
- class 1 = positive class = minority class
- class 0 = negative class = majority class

Confusion matrix:

    [[TN, FP],
     [FN, TP]]

The main metric is not plain accuracy. With class imbalance, a model can get
high accuracy by predicting only the majority class. So this module also
checks the majority baseline, balanced accuracy, and minority-class error.
"""

import numpy as np


def safe_division(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def format_confusion_matrix(confusion_matrix, class_labels=("0", "1")):
    """Return a readable table for a binary confusion matrix.

    Expected matrix layout:

        [[TN, FP],
         [FN, TP]]
    """
    matrix = np.asarray(confusion_matrix)

    if matrix.shape != (2, 2):
        raise ValueError("confusion_matrix must have shape (2, 2).")

    if len(class_labels) != 2:
        raise ValueError("class_labels must contain exactly two labels.")

    negative_label, positive_label = [str(label) for label in class_labels]
    headers = ["", f"Pred {negative_label}", f"Pred {positive_label}"]
    rows = [
        [f"True {negative_label}", str(matrix[0, 0]), str(matrix[0, 1])],
        [f"True {positive_label}", str(matrix[1, 0]), str(matrix[1, 1])],
    ]

    column_widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]

    def format_row(row):
        return "  ".join(
            value.rjust(column_widths[index]) for index, value in enumerate(row)
        )

    return "\n".join(
        [
            "Confusion matrix [[TN, FP], [FN, TP]]",
            format_row(headers),
            format_row(rows[0]),
            format_row(rows[1]),
        ]
    )


class BinaryClassImbalanceMetrics:
    """Metrics for binary labels where 1 is the minority class."""

    def __init__(self, y_true, y_pred):
        self.y_true = np.asarray(y_true).astype(int)
        self.y_pred = np.asarray(y_pred).astype(int)

        if len(self.y_true) != len(self.y_pred):
            raise ValueError("y_true and y_pred must have the same length.")

        self.tn = int(np.sum((self.y_true == 0) & (self.y_pred == 0)))
        self.fp = int(np.sum((self.y_true == 0) & (self.y_pred == 1)))
        self.fn = int(np.sum((self.y_true == 1) & (self.y_pred == 0)))
        self.tp = int(np.sum((self.y_true == 1) & (self.y_pred == 1)))

    @property
    def confusion_matrix(self):
        return np.array([[self.tn, self.fp], [self.fn, self.tp]])

    @property
    def total(self):
        return len(self.y_true)

    @property
    def majority_count(self):
        return self.tn + self.fp

    @property
    def minority_count(self):
        return self.tp + self.fn

    def accuracy(self):
        return safe_division(self.tp + self.tn, self.total)

    def majority_baseline_accuracy(self):
        return safe_division(self.majority_count, self.total)

    def accuracy_gain_over_majority(self):
        return self.accuracy() - self.majority_baseline_accuracy()

    def minority_recall(self):
        return safe_division(self.tp, self.tp + self.fn)

    def minority_error_rate(self):
        return safe_division(self.fn, self.tp + self.fn)

    def majority_recall(self):
        return safe_division(self.tn, self.tn + self.fp)

    def balanced_accuracy(self):
        return (self.minority_recall() + self.majority_recall()) / 2

    def pretty_print_confusion_matrix(self, class_labels=("0", "1")):
        """Print and return a readable table for a binary confusion matrix."""
        formatted_matrix = format_confusion_matrix(self.confusion_matrix, class_labels)
        print(formatted_matrix)
        return formatted_matrix

    def summary(self):
        return {
            "accuracy": self.accuracy(),
            "majority_baseline_accuracy": self.majority_baseline_accuracy(),
            "accuracy_gain_over_majority": self.accuracy_gain_over_majority(),
            "balanced_accuracy": self.balanced_accuracy(),
            "minority_recall": self.minority_recall(),
            "minority_error_rate": self.minority_error_rate(),
        }
