import pandas as pd
from logging import Logger
from ds_package_classifier.utils import metrics
from ds_fraud_detection_app.fraud_classifier import FraudClassifier


def calculate_trained_classifier_metrics(
    fraud_classifier: FraudClassifier, n_eval_folds: int
) -> pd.DataFrame:
    """
    Calculate cross-validation metrics for a trained fraud classifier at predefined thresholds.

    __PARAMETERS__
    fraud_classifier: FraudClassifier
        The trained classifier object containing threshold attributes
    n_eval_folds: int
        The number of folds to use for cross-validation

    __RETURNS__
    pd.DataFrame
        A DataFrame containing evaluation metrics (accuracy, precision, recall, F1) for each threshold
    """
    df_threshold_metrics = metrics.get_binary_thresholds_cross_val_prediction_metrics(
        fraud_classifier,
        thresholds=[
            fraud_classifier.medium_warning_level_threshold,
            fraud_classifier.high_warning_level_threshold,
        ],
        n_splits=n_eval_folds,
    )
    return df_threshold_metrics


def log_threshold_metrics(df_threshold_metrics: pd.DataFrame, logger: Logger) -> None:
    """
    Log threshold evaluation metrics row by row using the provided logger.

    __PARAMETERS__
    df_threshold_metrics: pd.DataFrame
        A DataFrame containing threshold evaluation metrics
    logger: Logger
        A logger instance used to output formatted metric information

    __RETURNS__
    None
    """
    for _, row in df_threshold_metrics.iterrows():
        logger.info(
            f"Threshold: {row['threshold']:.3f} | "
            f"Accuracy: {row['accuracy']:.3f} | "
            f"Precision: {row['precision']:.3f} | "
            f"Recall: {row['recall']:.3f} | "
            f"F1 Score: {row['f1']:.3f}"
        )

    return
