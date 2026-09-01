import pandas as pd
from typing import Optional
from logging import Logger
from copy import deepcopy
from datetime import datetime

from ds_package_gcp.bigquery import DataLoader
from ds_package_gcp.vertex_ai import (
    upload_model,
    load_model_artifact,
)
from ds_fraud_detection_app.fraud_classifier import FraudClassifier


# Load features from BigQuery
def load_bigquery_features(database: str, features_table: str) -> pd.DataFrame:
    """
    Load feature data from a BigQuery table.

    __PARAMETERS__
    database: str
        The name of the BigQuery database
    features_table: str
        The name of the table containing feature data

    __RETURNS__
    pd.DataFrame
        A DataFrame containing the downloaded feature data
    """
    data_loader = DataLoader()
    features_query = f"SELECT * FROM {database}.{features_table}"
    df_features = data_loader.run_query(features_query)
    return df_features


# Save metrics dataframe to BigQuery
def save_classifier_metrics(
    df_threshold_metrics: pd.DataFrame,
    fraud_classifier: FraudClassifier,
    datetime_tag: str,
    n_eval_folds: int,
    env: str = "dev",
) -> None:
    """
    Save model evaluation metrics to BigQuery.

    __PARAMETERS__
    df_threshold_metrics: pd.DataFrame
        DataFrame containing threshold-based evaluation metrics
    fraud_classifier: FraudClassifier
        The trained fraud classifier object
    datetime_tag: str
        Timestamp tag used to identify the training run
    n_eval_folds: int
        Number of evaluation folds used during cross-validation
    env: str
        The environment in which the pipeline is running (e.g., 'dev', 'main')

    __RETURNS__
    None
    """
    tmp_df_threshold_metrics = deepcopy(df_threshold_metrics)
    tmp_df_threshold_metrics["datetime_tag"] = datetime_tag
    tmp_df_threshold_metrics["n_eval_folds"] = n_eval_folds
    tmp_df_threshold_metrics["n_fraud_cases"] = (
        fraud_classifier.df[fraud_classifier.target_column] == 1
    ).sum()
    data_loader = DataLoader()
    data_loader.append_table(
        "ds_metrics_fraud",
        f"{env}_cross_val_predicition_metrics",
        tmp_df_threshold_metrics,
    )


# Save classifier to artifacts registry
def upload_model_to_vertex_ai(
    fraud_classifier: FraudClassifier, datetime_tag: str, config: dict, env: str = "dev"
) -> None:
    """
    Upload the trained fraud classifier to Vertex AI model registry.

    __PARAMETERS__
    fraud_classifier: FraudClassifier
        The trained fraud classifier object
    datetime_tag: str
        Timestamp tag used to version the model
    config: dict
        Configuration dictionary containing GCP and model registry settings
    env: str
        The environment in which the pipeline is running (e.g., 'dev', 'main')

    __RETURNS__
    None
    """
    upload_model(
        bucket_name=config["gcp"]["bucket_name"],
        parent_model=config["gcp"][env]["model_registry_model"],
        artifact_name=config["gcp"][env]["artifact_name"],
        model_name=config["gcp"][env]["display_name"],
        model=fraud_classifier,
        serving_container_image_uri=config["gcp"]["base_image_path"],
        version_tag=datetime_tag,
    )


# Load model artifact from GCP using Vertex AI model registry metadata
def download_model(config: dict, env: str = "dev") -> FraudClassifier:
    """
    Load latest fraud classifier model artifact from GCP using Vertex AI model registry metadata.

    __PARAMETERS__
    config: dict
        Configuration dictionary containing GCP model registry settings
    env: str
        The environment in which the pipeline is running (e.g., 'dev', 'main')

    __RETURNS__
    FraudClassifier
        The loaded fraud classifier object
    """
    fraud_classifier, version_tag = load_model_artifact(
        model_registry_model=config["gcp"][env]["model_registry_model"]
    )
    return fraud_classifier, version_tag


def create_all_bookings_fraud_predictions_table(
    df_features: pd.DataFrame,
    config: dict,
    fraud_classifier: FraudClassifier,
    y_pred_proba: list[float],
    include_target: bool = False,
) -> pd.DataFrame:
    """
    Construct a results table with prediction outputs and relevant features for all bookings.

    __PARAMETERS__
    df_features: pd.DataFrame
        DataFrame containing the original input data for prediction
    config: dict
        Configuration dictionary specifying feature groups used during training
    fraud_classifier: FraudClassifier
        The trained fraud classifier object containing feature metadata
    y_pred_proba: list[float]
        List of predicted fraud probabilities for each input record
    include_target: bool
        Boolean to signal whether to keep the target in the table
    __RETURNS__
    pd.DataFrame
        A DataFrame containing selected features and prediction outputs
    """

    # Get all features used during training
    booking_metadata = config["predictions"]["booking_metadata"]
    categorical_features = config["features"]["categorical_features"]
    binary_features = config["features"]["binary_features"]
    numeric_features = config["features"]["numeric_features"]
    top_binarized_multi_category_features = (
        fraud_classifier.top_binarized_multi_category_features
    )

    # Get target column if it is to be included
    target_column = []
    if include_target:
        target_column = [fraud_classifier.target_column]

    # Construct results table
    df_results = deepcopy(
        df_features[
            booking_metadata
            + categorical_features
            + binary_features
            + numeric_features
            + top_binarized_multi_category_features
            + target_column
        ]
    )

    # Generate fraud probability and warning levels columns
    df_results["predicted_fraud_probability"] = y_pred_proba

    # Generate a generic is_predicted_fraud boolean
    df_results["is_predicted_fraud"] = (
        df_results["predicted_fraud_probability"] >= 0.5
    ).astype(int)

    # Function to assign warning level based on fraud probability
    def _assign_warning_level(row, medium_threshold, high_threshold, proba_col):
        if row[proba_col] >= high_threshold:
            return "high"
        elif row[proba_col] >= medium_threshold:
            return "medium"
        else:
            return "low"

    df_results["warning_level"] = df_results.apply(
        _assign_warning_level,
        axis=1,
        args=(
            fraud_classifier.medium_warning_level_threshold,
            fraud_classifier.high_warning_level_threshold,
            "predicted_fraud_probability",
        ),
    )

    # Create datetime column of when predictions were made
    df_results["prediction_datetime"] = datetime.now()

    return df_results


def append_predictions_table(
    df_results: pd.DataFrame,
    logger: Logger,
    database_name: Optional[str],
    table_name: Optional[str],
    env: str = "dev",
    table_tag: Optional[str] = None,
) -> pd.DataFrame:
    """
    Append predictions results to table in BigQuery.

    __PARAMETERS__
    df_results: pd.DataFrame
        The DataFrame containing fraud prediction results to be saved.
    logger: Logger
        Logger instance used to log the save operation.
    database_name: Optional[str]
        Name of the BigQuery database where the table will be saved.
    table_name: Optional[str]
        Name of the BigQuery table to save the results to.
    env: str
        The environment in which the pipeline is running (e.g., 'dev', 'main')
    table_tag: Optional[str], default=None
        Optional tag to append to the table name for versioning or identification.

    __RETURNS__
    pd.DataFrame
        The same DataFrame that was saved to BigQuery.
    """

    data_loader = DataLoader()

    # Add env to table name if not main
    if not env == "main":
        table_name = f"{env}_{table_name}"

    # Add tag to table name if given
    if table_tag:
        table_name = f"{table_name}_{table_tag}"

    # Save the table to BigQuery
    logger.info(f"Saved table to {database_name}.{table_name}")
    data_loader.append_table(database_name, table_name, df_results)


def create_warnings_table(df_results: pd.DataFrame):
    """
    Create a table containing just the predicted fraudulent booking numbers and their prediction results

    __PARAMETERS__
    df_results: pd.DataFrame

    __RETURNS__
    pd.DataFrame
        The filtered warnings table
    """
    df_warnings = deepcopy(df_results)
    df_warnings = df_warnings[df_warnings["is_predicted_fraud"] == 1]
    return df_warnings


def append_warnings_table(
    df_results: pd.DataFrame,
    logger: Logger,
    database_name: Optional[str],
    table_name: Optional[str],
    env: str = "dev",
) -> pd.DataFrame:
    """
    Append predictions results to table in BigQuery.

    __PARAMETERS__
    df_results: pd.DataFrame
        The DataFrame containing fraud prediction results to be saved.
    logger: Logger
        Logger instance used to log the save operation.
    database_name: Optional[str]
        Name of the BigQuery database where the table will be saved.
    table_name: Optional[str]
        Name of the BigQuery table to save the results to.
    table_tag: Optional[str], default=None
        Optional tag to append to the table name for versioning or identification.
    env: str
        The environment in which the pipeline is running (e.g., 'dev', 'main')

    __RETURNS__
    pd.DataFrame
        The same DataFrame that was saved to BigQuery.
    """

    data_loader = DataLoader()

    # Save the table to BigQuery
    logger.info(f"Save warning bookings table to {database_name}.{table_name}")
    data_loader.append_table(database_name, table_name, df_results)
