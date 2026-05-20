import sys
from src.utils.exception import CustomException
from src.utils.logger import get_logger

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

logger = get_logger(__name__)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Trains multiple classifiers with hyperparameter tuning via GridSearchCV
    and returns a full performance report + trained model objects.

    Args:
        X_train, y_train : training features and labels
        X_test,  y_test  : test features and labels
        models (dict)    : { model_name: sklearn_estimator }
        params (dict)    : { model_name: param_grid }

    Returns:
        report (dict)        : per-model metrics
        trained_models (dict): per-model fitted estimator
        best_model_name (str): name of the best model by test F1
    """
    try:
        report = {}
        trained_models = {}

        # StratifiedKFold preserves class balance in each fold — better than plain cv=5
        # for binary sentiment (pos/neg may not be perfectly balanced)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            param_grid = params.get(model_name, {})

            # ── Hyperparameter search ─────────────────────────────────────────
            if param_grid:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=cv,
                    scoring='f1',           # optimise for F1, not accuracy
                    n_jobs=-1,              # use all CPU cores
                    verbose=1
                )
                gs.fit(X_train, y_train)
                best_params = gs.best_params_
                logger.info(f"[{model_name}] Best params: {best_params}")
                model.set_params(**best_params)
            else:
                best_params = {}
                logger.info(f"[{model_name}] No param grid — using defaults.")

            # ── Train with best params ────────────────────────────────────────
            model.fit(X_train, y_train)

            # ── Predict ───────────────────────────────────────────────────────
            y_train_pred = model.predict(X_train)
            y_test_pred  = model.predict(X_test)

            # ── Metrics (classification, not regression) ──────────────────────
            metrics = {
                'best_params':        best_params,

                # Test metrics (what actually matters)
                'test_accuracy':      accuracy_score(y_test, y_test_pred),
                'test_precision':     precision_score(y_test, y_test_pred, zero_division=0),
                'test_recall':        recall_score(y_test, y_test_pred, zero_division=0),
                'test_f1':            f1_score(y_test, y_test_pred, zero_division=0),

                # Train metrics (to detect overfitting)
                'train_accuracy':     accuracy_score(y_train, y_train_pred),
                'train_f1':           f1_score(y_train, y_train_pred, zero_division=0),

                # Overfitting gap — large gap means overfit
                'overfit_gap':        round(
                    accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_test_pred), 4
                ),

                # Full breakdown per class
                'classification_report': classification_report(
                    y_test, y_test_pred, target_names=['negative', 'positive']
                ),
                'confusion_matrix':   confusion_matrix(y_test, y_test_pred).tolist(),
            }

            report[model_name] = metrics
            trained_models[model_name] = model

            # ── Log summary ───────────────────────────────────────────────────
            logger.info(
                f"[{model_name}] "
                f"Test Acc: {metrics['test_accuracy']:.4f} | "
                f"Test F1: {metrics['test_f1']:.4f} | "
                f"Overfit gap: {metrics['overfit_gap']:.4f}"
            )
            print(f"\n{'='*55}")
            print(f"  {model_name}")
            print(f"{'='*55}")
            print(f"  Test  Accuracy : {metrics['test_accuracy']:.4f}")
            print(f"  Test  F1       : {metrics['test_f1']:.4f}")
            print(f"  Test  Precision: {metrics['test_precision']:.4f}")
            print(f"  Test  Recall   : {metrics['test_recall']:.4f}")
            print(f"  Train Accuracy : {metrics['train_accuracy']:.4f}  (gap: {metrics['overfit_gap']:+.4f})")
            print(f"\n  Classification Report:\n{metrics['classification_report']}")
            print(f"  Confusion Matrix: {metrics['confusion_matrix']}")

        # ── Pick best model by test F1 ────────────────────────────────────────
        best_model_name = max(report, key=lambda m: report[m]['test_f1'])
        logger.info(f"Best model: {best_model_name} | F1: {report[best_model_name]['test_f1']:.4f}")
        print(f"\n>>> Best Model: {best_model_name}  (F1 = {report[best_model_name]['test_f1']:.4f})")

        return report, trained_models, best_model_name

    except Exception as e:
        raise CustomException(e, sys)
