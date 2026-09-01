import pytest
import pandas as pd

from ds_fraud_detection_app.utils.feature_engineering import (
    engineer_is_online_bank_used,
    engineer_is_non_card_payment_method_used,
    engineer_is_residence_bank_country_matched,
    engineer_is_residence_traffic_sessions_country_matched,
    build_residence_country_code,
)


# df with residence country codes instead of country names to test each individual feature engineering function
@pytest.fixture
def sample_df_residence_country_codes():
    df_test = pd.DataFrame(
        {
            "bank_names": [
                "wise",
                "bbva",
                "revolut",
                "hsbc",
                "nab",
            ],
            "payment_methods": ["paypal", "visa", "mc", "visa", "visa"],
            "residence_country_code": ["uk", "es", "fr", "uk", "au"],
            "bank_country_codes": [
                "uk",
                "es",
                "uk",
                "de",
                "au",
            ],
            "traffic_sessions_countries": [
                ["United Kingdom", "Spain"],
                ["Spain"],
                ["France"],
                ["Germany", "Australia"],
                ["Australia"],
            ],
        }
    )
    return df_test


@pytest.fixture
def online_banks():
    return ["wise", "revolut", "monzo"]


@pytest.fixture
def non_card_payment_methods():
    return ["paypal", "directEbanking"]


def test_engineer_is_online_bank_used(sample_df_residence_country_codes, online_banks):

    # Assert extra column added
    n_original_columns = len(sample_df_residence_country_codes.columns)
    df_result = engineer_is_online_bank_used(
        sample_df_residence_country_codes, online_banks
    )
    assert len(df_result.columns) == n_original_columns + 1

    # Assert column exists
    assert "is_online_bank_used" in df_result.columns

    # Assert all values are either 0 or 1
    valid_values = {0, 1}
    assert set(df_result["is_online_bank_used"].unique()).issubset(valid_values)

    # Assert correct number of flagged rows
    df_result["is_online_bank_used"].sum() == 2


def test_engineer_is_non_card_payment_method_used(
    sample_df_residence_country_codes, non_card_payment_methods
):

    # Assert extra column added
    n_original_columns = len(sample_df_residence_country_codes.columns)
    df_result = engineer_is_non_card_payment_method_used(
        sample_df_residence_country_codes, non_card_payment_methods
    )
    assert len(df_result.columns) == n_original_columns + 1

    # Assert column exists
    assert "is_non_card_payment_method_used" in df_result.columns

    # Assert all values are either 0 or 1
    valid_values = {0, 1}
    assert set(df_result["is_non_card_payment_method_used"].unique()).issubset(
        valid_values
    )

    # Assert correct number of flagged rows
    df_result["is_non_card_payment_method_used"].sum() == 1


def test_engineer_is_residence_bank_country_matched(sample_df_residence_country_codes):

    # Assert extra column added
    n_original_columns = len(sample_df_residence_country_codes.columns)
    df_result = engineer_is_residence_bank_country_matched(
        sample_df_residence_country_codes
    )
    assert len(df_result.columns) == n_original_columns + 1

    # Assert column exists
    assert "is_residence_bank_country_matched" in df_result.columns

    # Assert all values are either 0 or 1
    valid_values = {0, 1}
    assert set(df_result["is_residence_bank_country_matched"].unique()).issubset(
        valid_values
    )

    # Assert correct number of flagged rows
    df_result["is_residence_bank_country_matched"].sum() == 3


def test_engineer_is_residence_traffic_sessions_country_matched(
    sample_df_residence_country_codes,
):

    # Assert extra column added
    n_original_columns = len(sample_df_residence_country_codes.columns)
    df_result = engineer_is_residence_traffic_sessions_country_matched(
        sample_df_residence_country_codes
    )
    assert len(df_result.columns) == n_original_columns + 1

    # Assert column exists
    assert "is_residence_traffic_sessions_country_matched" in df_result.columns

    # Assert all values are either 0 or 1
    valid_values = {0, 1}
    assert set(
        df_result["is_residence_traffic_sessions_country_matched"].unique()
    ).issubset(valid_values)

    # Assert correct number of flagged rows
    df_result["is_residence_traffic_sessions_country_matched"].sum() == 4


@pytest.fixture
def sample_df_residence_country_names():
    df_test = pd.DataFrame(
        {
            "bank_names": [
                "wise",
                "bbva",
                "revolut",
                "hsbc",
                "nab",
            ],
            "payment_methods": ["paypal", "visa", "mc", "visa", "visa"],
            "residence_country": [
                "United Kingdom",
                "Spain",
                "France",
                "United Kingdom",
                "Australia",
            ],
            "bank_country_codes": [
                "gb",
                "es",
                "gb",
                "de",
                "au",
            ],
            "traffic_sessions_countries": [
                ["United Kingdom", "Spain"],
                ["Spain"],
                ["France"],
                ["Germany", "Australia"],
                ["Australia"],
            ],
        }
    )
    return df_test


def test_build_residence_country_code(sample_df_residence_country_names):
    # Assert all new feature columns added
    n_original_columns = len(sample_df_residence_country_names.columns)
    df_result = build_residence_country_code(sample_df_residence_country_names)

    assert len(df_result.columns) == n_original_columns + 1

    # Assert column exists
    assert "residence_country_code" in df_result.columns

    # Assert column exists
    assert "gb" in df_result["residence_country_code"].values
