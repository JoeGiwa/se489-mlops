from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(preds, labels):
    """
    Calculates evaluation metrics given predictions and ground truth labels.
    """
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1_score": f1}
