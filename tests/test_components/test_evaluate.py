import pytest
from sklearn.datasets import load_breast_cancer
import pandas as pd
from xgboost import XGBClassifier

from ds_fraud_detection_app.components.evaluate import (
    calculate_trained_classifier_metrics,
)
from ds_fraud_detection_app.fraud_classifier import FraudClassifier


# Binary class data sample from sklearn breast_cancer dataset
@pytest.fixture
def sample_df_binary_class():
    data = load_breast_cancer()
    df_test = pd.DataFrame(data.data, columns=data.feature_names)
    df_test["target"] = data.target
    return df_test


@pytest.fixture
def target():
    return "target"


# train a simple fraud classifier to use in tests
@pytest.fixture
def trained_fraud_classifier(sample_df_binary_class, target):

    # Define the classifier
    xgb_model = XGBClassifier()
    fraud_classifier = FraudClassifier(
        sample_df_binary_class, target, xgb_model, medium_warning_level_threshold=0.5
    )

    # Train the classifier
    fraud_classifier.train()

    # manually set high_warning_level_threshold
    fraud_classifier.high_warning_level_threshold = 0.9
    return fraud_classifier


def test_calculate_trained_classifier_metrics(trained_fraud_classifier):
    df_threshold_metrics = calculate_trained_classifier_metrics(
        trained_fraud_classifier, 3
    )

    # Assert correct column names
    assert set(df_threshold_metrics.columns) == {
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
    }

    # Assert correct shape
    assert df_threshold_metrics.shape[0] == 2
    assert df_threshold_metrics.shape[1] == 5

    # Assert all values are floats
    all(df_threshold_metrics.dtypes == float)
