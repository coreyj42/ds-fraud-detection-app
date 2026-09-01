import pytest
import pandas as pd
from datetime import datetime, date, timedelta

from ds_fraud_detection_app.utils.utils import load_config
from ds_fraud_detection_app.components.preprocess import (
    filter_rows,
    preprocess_training_features,
    convert_data_types,
    generate_missing_binarized_feature_columns,
)


# DataFrame with values to be removed before modelling
@pytest.fixture
def sample_df_filter():
    df_test = pd.DataFrame(
        {
            "total_nights": [1, 8, 7, 13, 3, 8, 7, 13, 3, 3],
            "customer_age_at_booking_creation": [
                23,
                123,
                22,
                99,
                45,
                13,
                22,
                23,
                44,
                33,
            ],
            "booking_creation_date": [
                date(2025, 4, 1),
                date(2024, 11, 5),
                date(2022, 10, 10),
                date(2024, 10, 15),
                date(2025, 2, 1),
                date(2025, 9, 1),
                date(2024, 1, 5),
                date(2025, 3, 2),
                date(2024, 11, 15),
                date(2023, 1, 1),
            ],
            "pickup_date": [
                date(2025, 4, 1),
                date(2024, 11, 5),
                date(2022, 10, 10),
                date.today(),
                date(2025, 2, 1),
                date(2025, 9, 1),
                date(2024, 1, 5),
                date(2025, 3, 2),
                date(2024, 11, 15),
                date(2023, 1, 1),
            ],
            "outlier_test_column": [
                0.92,
                1.12,
                0.94,
                1.02,
                99.0,
                0.95,
                1.11,
                0.92,
                1.09,
                -99.0,
            ],
            "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
    )
    return df_test


@pytest.fixture
def config_filter_features():
    return load_config("config_tests.yaml")["filter_features"]


def test_filter_rows(sample_df_filter, config_filter_features):
    # Assert correct number of rows removed
    n_original_rows = len(sample_df_filter)
    df_result = filter_rows(sample_df_filter, config_filter_features)
    assert len(df_result) == n_original_rows - 5

    # Assert correct values have been removed
    assert 1 not in df_result["total_nights"].values
    assert 123 not in df_result["customer_age_at_booking_creation"].values
    assert date(2022, 10, 10) not in df_result["booking_creation_date"].values
    assert date.today() not in df_result["pickup_date"].values
    assert 99.0 not in df_result["outlier_test_column"].values


# DataFrame with all feature types to be preprocessed
@pytest.fixture
def sample_df_preprocess():
    df_test = pd.DataFrame(
        {
            "numeric": [1, 8, 7, 13, 3, 8, 7, 13, 3, 3],
            "categorical": [
                "France",
                "France",
                "United Kingdom",
                "France",
                None,
                "France",
                "France",
                "United Kingdom",
                "France",
                "France",
            ],
            "binary": [0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
            "multi_categorical": [
                ["mc", "visa"],
                ["visa"],
                ["visa"],
                ["mc", "visa"],
                ["mc"],
                ["visa"],
                ["mc"],
                ["mc"],
                ["mc"],
                ["mc"],
            ],
            "unused_column": [23, 123, 22, 99, 45, 13, 22, 23, 44, 33],
            "metadata_column": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
    )
    return df_test


@pytest.fixture
def config_preprocess():
    return load_config("config_tests.yaml")["preprocess"]


def test_preprocess_training_features(sample_df_preprocess, config_preprocess):
    df_result, expanded_multi_categorical_features = preprocess_training_features(
        sample_df_preprocess, config_preprocess
    )

    # Assert column undefined for training is removed
    assert "unused_column" not in df_result.columns

    # Assert "missing" added to replace None values
    assert "missing" in df_result["categorical"].values

    # Assert expanded_multi_categorical_features is correct
    set(expanded_multi_categorical_features) == {
        "multi_categorical_mc",
        "multi_categorical_visa",
    }

    # Asset  expanded_multi_categorical_features columns are in updated DataFrame
    assert set(expanded_multi_categorical_features).issubset(set(df_result.columns))

    # Assert metadata column is kept in table despite being unused for training
    assert "metadata_column" in df_result.columns


# DataFrame with datetime values to covert to dates
@pytest.fixture
def sample_df_datetimes():

    df_test = pd.DataFrame(
        {
            "booking_number": [123, 456, 789],
            "booking_creation_date": [
                datetime(2025, 4, 1),
                datetime(2024, 11, 5),
                datetime(2022, 10, 10),
            ],
            "pickup_date": [
                datetime(2025, 4, 2),
                datetime(2024, 11, 6),
                datetime(2022, 10, 11),
            ],
            "is_promotion_used": [True, False, True],
            "is_fraud": [False, False, True],
            "is_customer_details_changed_3_days_before_pickup": [True, False, False],
            "is_tracked_traffic_sessions": [True, True, False],
        }
    )

    return df_test


def test_convert_data_types(sample_df_datetimes):
    df_result = convert_data_types(sample_df_datetimes)

    # Assert booking_number has been converted to string
    assert df_result["booking_number"].dtype == object
    assert all(isinstance(x, str) for x in df_result["booking_number"])

    # Assert datetime columns have been converted to dates
    assert all(
        isinstance(d, date) and not isinstance(d, datetime)
        for d in df_result["booking_creation_date"]
    )
    assert all(
        isinstance(d, date) and not isinstance(d, datetime)
        for d in df_result["pickup_date"]
    )

    # Assert boolean columns converted to integers
    int_columns = [
        "is_promotion_used",
        "is_fraud",
        "is_customer_details_changed_3_days_before_pickup",
        "is_tracked_traffic_sessions",
    ]
    for col in int_columns:
        assert df_result[col].dtype == "int64"
        assert set(df_result[col].unique()).issubset({0, 1})


# DataFrame with booking_creation_date from today
@pytest.fixture
def sample_df_updated():
    yesterday = date.today() - timedelta(days=1)
    df_test = pd.DataFrame(
        {"booking_creation_date": [date(2025, 4, 1), date(2024, 11, 5), yesterday]}
    )
    return df_test


# DataFrame without booking_creation_date from today
@pytest.fixture
def sample_df_not_updated():
    two_days_ago = date.today() - timedelta(days=2)
    df_test = pd.DataFrame(
        {"booking_creation_date": [date(2025, 4, 1), date(2024, 11, 5), two_days_ago]}
    )
    return df_test


# DataFrame with datetime values to covert to dates
@pytest.fixture
def sample_df_binarized():
    df_test = pd.DataFrame(
        {
            "binarized_feature_a": [0, 0, 1],
            "binarized_feature_b": [1, 0, 1],
        }
    )
    return df_test


@pytest.fixture
def binarized_feature_column_names():
    return ["binarized_feature_a", "binarized_feature_b", "binarized_feature_c"]


# Test adding missing binarized columns that were used in training data and needed for predictions
def test_generate_missing_binarized_feature_columns(
    sample_df_binarized, binarized_feature_column_names
):
    df_result = generate_missing_binarized_feature_columns(
        sample_df_binarized, binarized_feature_column_names
    )

    # Assert that binarized_feature_c was generated and contained all zeros
    assert "binarized_feature_c" in df_result.columns
    assert (df_result["binarized_feature_c"] == 0).all()
