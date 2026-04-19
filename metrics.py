import numpy as np


def safe_division(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def get_counts(matrix):
    tp, fp = matrix[0]
    fn, tn = matrix[1]
    return tp, fp, fn, tn


def confusion_matrix(y_pred, y_true):
    """
    Returns the confusion matrix given the true and predicted values
      1   0
    1 TP  FP
    0 FN  TN
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(len(y_true)):
        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            fp += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            fn += 1
        elif y_pred[i] == 0 and y_true[i] == 0:
            tn += 1

    return np.array([[tp, fp], [fn, tn]])


def accuracy(matrix):
    """
    accuracy: (TP + TN)/ (TP + TN + FP + FN)
    """
    tp, fp, fn, tn = get_counts(matrix)
    return safe_division(tp + tn, tp + fp + fn + tn)


def precision(matrix):
    """
    precision: TP / (TP+FP)
    """
    tp, fp, _, _ = get_counts(matrix)
    return safe_division(tp, tp + fp)


def recall(matrix):
    """
    recall: TP / (TP+FN)
    """
    tp, _, fn, _ = get_counts(matrix)
    return safe_division(tp, tp + fn)


def f1_score(matrix):
    """
    f1-score: 2 * precision * recall / (precision + recall)
    """
    p = precision(matrix)
    r = recall(matrix)
    return safe_division(2 * p * r, p + r)


def balanced_accuracy(matrix):
    """
    balanced accuracy: (recall + specificity) / 2
    """
    _, fp, _, tn = get_counts(matrix)
    recall_value = recall(matrix)
    specificity = safe_division(tn, tn + fp)
    return (recall_value + specificity) / 2
