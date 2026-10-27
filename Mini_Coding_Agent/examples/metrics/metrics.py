"""Small classification metrics library used by the teaching demo."""


def precision_recall(y_true, y_pred):
    """Return precision and recall for binary labels where 1 is positive."""
    tp = sum(actual == 1 and predicted == 1 for actual, predicted in zip(y_true, y_pred))
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in zip(y_true, y_pred))
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in zip(y_true, y_pred))
    return tp / (tp + fp), tp / (tp + fn)
