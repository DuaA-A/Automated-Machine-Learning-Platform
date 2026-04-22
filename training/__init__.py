from .classification import train_classification
from .regression import train_regression
from .clustering import train_clustering


def train_and_evaluate(X, y, task_type: str, algorithm_choice: str = "AutoML (Find Best Model)"):
    """
    Route to the correct training module based on task_type.
    Returns (best_model, results, prep_info).
    """
    if task_type == "Classification":
        return train_classification(X, y, algorithm_choice)
    elif task_type == "Regression":
        return train_regression(X, y, algorithm_choice)
    elif task_type == "Clustering":
        return train_clustering(X, algorithm_choice)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
