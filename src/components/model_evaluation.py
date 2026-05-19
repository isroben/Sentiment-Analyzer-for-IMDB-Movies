import sys
from src.utils.exception import CustomException
from src.utils.logger import get_logger

from sklearn.metrics import {
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confustion_matrix
}
from sklearn.model_selection import GridSearchCV, StratifiedKFold

logger = get_logger(__name__)

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        trained_models = {}

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            params_grid = params.get(model_name, {})

            if params_grid:
                gs = GridSearchCV(
                    estimator=model,
                    params_grid = params_grid,
                    cv = cv,
                    scoring='f1',
                    n_jobs=1,
                    verbose=1
                )

                gs.fit(X_train, y_train)
                best_params = gs.best_params_
                logger.info(f"[{model_name}] Best params: {best_params}")
                model.set_params(**best_params)

            else:
                best_params = {}
                logger.info(f"[{model_name}] No param grid - using defaults.")

            model.fit(X_train, y_train)