"""Prediction model constructors."""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SOURCE_LOGISTIC_GRID = {"classifier__C": [0.001, 0.01, 0.1, 1.0]}
SOURCE_HIST_GRADIENT_BOOSTING_GRID = {
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.1],
}


def logistic_model(*, random_state: int = 0) -> LogisticRegression:
    """Return the registered regularized binary classifier."""
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
        solver="lbfgs",
    )


def source_logistic_pipeline(*, random_state: int = 0) -> Pipeline:
    """Create the source-only L2 logistic model with train-fitted scaling."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def source_hist_gradient_boosting(
    *, random_state: int = 0
) -> HistGradientBoostingClassifier:
    """Create the registered source-only histogram gradient boosting model."""
    return HistGradientBoostingClassifier(
        max_iter=200,
        early_stopping=True,
        l2_regularization=1.0,
        random_state=random_state,
    )
