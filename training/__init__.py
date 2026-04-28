from .classification import train_classification
from .regression import train_regression
from .clustering import train_clustering


def train_and_evaluate(X, y, task_type: str):
    if task_type == "Classification":
        return train_classification(X, y)
    elif task_type == "Regression":
        return train_regression(X, y)
    elif task_type == "Clustering":
        return train_clustering(X)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
