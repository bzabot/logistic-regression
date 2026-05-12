import numpy as np


def safe_division(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def get_counts(matrix):
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    return tp, fp, fn, tn


def confusion_matrix(y_true, y_pred):
    """
    Return the binary confusion matrix in the conventional sklearn layout.

           Pred 0  Pred 1
    True 0   TN      FP
    True 1   FN      TP
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            tn += 1

    return np.array([[tn, fp], [fn, tp]])


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
    recall_value = recall(matrix)
    specificity_value = specificity(matrix)
    return (recall_value + specificity_value) / 2


def specificity(matrix):
    """
    specificity / true negative rate: TN / (TN+FP)
    """
    _, fp, _, tn = get_counts(matrix)
    return safe_division(tn, tn + fp)


def false_positive_rate(matrix):
    """
    false positive rate: FP / (FP+TN)
    """
    _, fp, _, tn = get_counts(matrix)
    return safe_division(fp, fp + tn)
