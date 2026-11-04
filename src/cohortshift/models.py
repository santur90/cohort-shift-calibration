"""Prediction model constructors."""

from sklearn.linear_model import LogisticRegression


def logistic_model(*, random_state: int = 0) -> LogisticRegression:
    """Return the registered regularized binary classifier."""
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
        solver="lbfgs",
    )
