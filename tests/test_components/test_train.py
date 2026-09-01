import pytest
import pandas as pd

from ds_fraud_detection_app.utils.utils import load_config
from ds_fraud_detection_app.components.train import (
    get_top_feature_categories,
    train_final_model,
    calculate_high_warning_threshold,
)


# binary class DataFrame with all feature types to be preprocessed
@pytest.fixture
def sample_df():
    df_test = pd.DataFrame(
        {
            "numeric": [1, 8, 7, 13, 3, 192, 116, 131, 113, 321],
            "categorical": [
                "France",
                "France",
                "France",
                "France",
                "France",
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
            ],
            "binary": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "multi_categorical_paypal": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "multi_categorical_visa": [1, 1, 1, 0, 1, 0, 0, 1, 0, 0],
            "multi_categorical_mc": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )
    return df_test


@pytest.fixture
def target():
    return "target"


# Mimic saved column names generated from applying binarization
@pytest.fixture
def expanded_multi_categorical_features():
    return [
        "multi_categorical_paypal",
        "multi_categorical_visa",
        "multi_categorical_mc",
    ]


# Get config for train tests
@pytest.fixture
def config_train():
    return load_config("config_tests.yaml")["train"]


# Test calculating the top categorical feature categories
def test_get_top_feature_categories(
    sample_df, expanded_multi_categorical_features, config_train
):
    top_categories, top_binarized_multi_category_features = get_top_feature_categories(
        sample_df,
        config_train,
        expanded_multi_categorical_features=expanded_multi_categorical_features,
        imbalance_ratio=1.0,
    )

    # Asset returned lists are of correct length
    assert len(top_categories) == 1
    assert len(top_categories[0]) == 2
    assert len(top_binarized_multi_category_features) == 1

    # Assert returned lists contain correct top categories
    assert set(top_categories[0]) == {"France", "United Kingdom"}
    assert top_binarized_multi_category_features == ["multi_categorical_paypal"]


# Test training the final model and calculating the high warning threshold
# Tested together as one step is needed before the other
def test_train_final_model_and_high_warning_threshold(sample_df, config_train):

    # Train classifier
    test_classifier = train_final_model(
        sample_df,
        config_train,
        top_categories=[["france"]],
        top_binarized_multi_category_features=["multi_categorical_paypal"],
        imbalance_ratio=1.0,
    )

    # Make predictions
    test_predictions = test_classifier.predict(sample_df)

    # Assert predictions are as expected
    assert len(test_predictions) == 10
    assert test_predictions[0] == 0

    high_warning_level_threshold = calculate_high_warning_threshold(
        test_classifier, config_train, n_eval_folds=3
    )

    # Assert correct type is returned
    assert isinstance(high_warning_level_threshold, float)
