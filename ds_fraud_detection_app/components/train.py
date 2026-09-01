from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import pandas as pd

from ds_package_classifier.utils import xgboost_utils, metrics
from ds_fraud_detection_app.fraud_classifier import FraudClassifier


def get_top_feature_categories(
    df_features: pd.DataFrame,
    config: dict,
    expanded_multi_categorical_features: list[str],
    imbalance_ratio: float,
) -> tuple[list[list[str]], list[str]]:
    """
    Train an initial classifier to extract the most important categorical and multi-categorical features.

    __PARAMETERS__
    df_features: pd.DataFrame
        The input DataFrame containing training features
    config: dict
        Configuration dictionary containing feature definitions and model parameters
    expanded_multi_categorical_features: list[str]
        List of binarized multi-categorical feature names
    imbalance_ratio: float
        Ratio used to scale positive class weight for imbalanced classification

    __RETURNS__
    tuple[list[list[str]], list[str]]
        A tuple containing:
        - A list of top categories for each categorical feature
        - A list of top binarized multi-categorical feature names
    """

    # Extract training features for each value type
    categorical_features = config["features"]["categorical_features"]
    binary_features = config["features"]["binary_features"]
    numeric_features = config["features"]["numeric_features"]

    # Define pipelines used in column transformer
    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # Build column transformer used to preprocess features for the model
    column_transformer = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
            ("bin", "passthrough", binary_features),
            ("multi_cat", "passthrough", expanded_multi_categorical_features),
        ]
    )
    column_transformer.set_output(transform="pandas")

    # Build model hyperparameters
    hyperparams = config["model"]["hyperparameters"]
    hyperparams["scale_pos_weight"] = imbalance_ratio

    # Definer and train the model
    xgb_model = XGBClassifier(**hyperparams)
    tmp_fraud_classifier = FraudClassifier(
        df_features,
        config["data"]["target_column"],
        xgb_model,
        column_transformer=column_transformer,
    )
    tmp_fraud_classifier.train()

    # Extract the caregorical features that were most often used for splits during training
    top_categories = xgboost_utils.extract_most_used_categorical_features_from_pipeline(
        tmp_fraud_classifier,
        categorical_features,
        config["model"]["feature_selection"]["top_n_categorical"],
        "cat",
    )

    # Extract the binarized multi categorical features that were most often used for splits during training
    top_binarized_multi_category_features = (
        xgboost_utils.extract_most_used_multi_categorical_features_from_pipeline(
            tmp_fraud_classifier,
            config["model"]["feature_selection"]["top_n_multi_categorical"],
            multi_categorical_transformer_name="multi_cat",
        )
    )

    return top_categories, top_binarized_multi_category_features


def train_final_model(
    df_features: pd.DataFrame,
    config: dict,
    top_categories: list[list[str]],
    top_binarized_multi_category_features: list[str],
    imbalance_ratio: float,
) -> FraudClassifier:
    """
    Train the final fraud classifier using selected top features and return the trained model.

    __PARAMETERS__
    df_features: pd.DataFrame
        The input DataFrame containing training features
    config: dict
        Configuration dictionary containing feature definitions and model parameters
    top_categories: list[list[str]]
        List of top categories for each categorical feature
    top_binarized_multi_category_features: list[str]
        List of top binarized multi-categorical feature names
    imbalance_ratio: float
        Ratio used to scale positive class weight for imbalanced classification

    __RETURNS__
    FraudClassifier
        The trained fraud classifier object
    """

    # Extract training features for each value type
    categorical_features = config["features"]["categorical_features"]
    binary_features = config["features"]["binary_features"]
    numeric_features = config["features"]["numeric_features"]

    # Define pipelines used in column transformer
    categorical_transformer = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    categories=top_categories,
                ),
            )
        ]
    )
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # Build column transformer used to preprocess features for the model
    column_transformer = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
            ("bin", "passthrough", binary_features),
            ("multi_cat", "passthrough", top_binarized_multi_category_features),
        ]
    )
    column_transformer.set_output(transform="pandas")

    # Build model hyperparameters
    hyperparams = config["model"]["hyperparameters"]
    hyperparams["scale_pos_weight"] = imbalance_ratio

    # Definer and train the model
    xgb_model = XGBClassifier(**hyperparams)
    fraud_classifier = FraudClassifier(
        df_features,
        config["data"]["target_column"],
        xgb_model,
        column_transformer=column_transformer,
    )
    fraud_classifier.train()

    # Store top_binarized_multi_category_features as a class variable to be saved with the model for use during prediction
    fraud_classifier.top_binarized_multi_category_features = (
        top_binarized_multi_category_features
    )

    return fraud_classifier


def calculate_high_warning_threshold(
    fraud_classifier: FraudClassifier, config: dict, n_eval_folds: int
) -> float:
    """
    Calculate the high warning threshold for the fraud classifier.

    __PARAMETERS__
    fraud_classifier: FraudClassifier
        The trained fraud classifier object
    config: dict
        Configuration dictionary containing threshold settings
    n_eval_folds: int
        Number of folds to use for cross-validation

    __RETURNS__
    float
        The calculated high warning threshold value
    """

    # Define the beta values to test in f-beta metric calculations
    high_warning_level_beta = config["thresholds"]["high_warning_level_beta"]
    beta_values_to_test = [high_warning_level_beta]

    # Calculate the threshold that maximizes the f-beta metric
    best_thresholds = metrics.get_top_binary_thresholds(
        fraud_classifier, beta_values=beta_values_to_test, n_splits=n_eval_folds
    )

    # Set the high warning threshold for the model
    high_warning_level_threshold = best_thresholds[f"fbeta_{high_warning_level_beta}"]

    return high_warning_level_threshold
