import pandas as pd
from datetime import date, timedelta

from ds_package_classifier.utils import cleaning, preprocessing
from ds_package_gcp.bigquery import DataLoader


# Error to raise if data has not been updated
class StaleDataError(Exception):
    """Raised when the latest update date is older than expected."""

    pass


def filter_rows(df_features: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Filter and clean the input DataFrame by applying business rules and removing outliers.

    __PARAMETERS__
    df_features: pd.DataFrame
        The input DataFrame containing raw booking-level features
    config: dict
        Configuration dictionary containing filtering parameters and feature definitions

    __RETURNS__
    pd.DataFrame
        The filtered and cleaned DataFrame
    """

    # Filter rows with total_nights greater than 1 and customer age less than 100
    df_features = df_features[df_features["total_nights"] > 1]
    df_features = df_features[df_features["customer_age_at_booking_creation"] < 100]

    # Filter bookings with a pickup date before today
    today = date.today()
    df_features = df_features[df_features["pickup_date"] < today]

    # Remove bookings created before the cutoff year
    cutoff_year = int(config["data"]["cutoff_year"])
    cutoff_date = date(cutoff_year, 1, 1)
    df_features = df_features[df_features["booking_creation_date"] >= cutoff_date]

    # Drop outliers based on IQR method
    df_features = cleaning.drop_outliers_iqr(
        df=df_features,
        target_column=config["data"]["target_column"],
        outlier_columns=config["features"]["outlier_removal_features"],
        multiplier=3.0,
        withhold_classes=[1],
    )

    return df_features


def preprocess_training_features(
    df_features: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, list[str]]:
    """
    Preprocess the training features by selecting relevant columns, handling missing values,
    and binarizing multi-categorical features.

    __PARAMETERS__
    df_features: pd.DataFrame
        The input DataFrame containing filtered booking-level features
    config: dict
        Configuration dictionary containing feature definitions

    __RETURNS__
    tuple[pd.DataFrame, list[str]]
        A tuple containing:
        - The preprocessed DataFrame
        - A list of expanded multi-categorical feature names
    """

    # Extract feature types from config
    categorical_features = config["features"]["categorical_features"]
    binary_features = config["features"]["binary_features"]
    numeric_features = config["features"]["numeric_features"]
    multi_categorical_features = config["features"]["multi_categorical_features"]

    # Include extra columns not needed for training
    extra_columns = config["predictions"]["booking_metadata"]

    # Select training features and target column
    training_features = (
        extra_columns
        + categorical_features
        + binary_features
        + numeric_features
        + multi_categorical_features
    )
    df_features = df_features[training_features + [config["data"]["target_column"]]]

    # Replace missing values in categorical columns with "missing"
    df_features.loc[:, categorical_features] = df_features[categorical_features].fillna(
        "missing"
    )

    # Binarize list features and get new feature names
    df_features, expanded_multi_categorical_features = (
        preprocessing.binarize_list_features(df_features, multi_categorical_features)
    )

    return df_features, expanded_multi_categorical_features


# Convert columns to correct data types needed for the model
def convert_data_types(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure feature columns are correct types.

    __PARAMETERS__
    df_features: pd.DataFrame
        The input DataFrame containing booking-level features

    __RETURNS__
    pd.DataFrame
        The DataFrame with feature columns converted to correct data types
    """

    # Convert columns to correct data types
    df_features["booking_number"] = df_features["booking_number"].astype(str)
    df_features["booking_creation_date"] = df_features["booking_creation_date"].dt.date
    df_features["pickup_date"] = df_features["pickup_date"].dt.date
    df_features["is_promotion_used"] = df_features["is_promotion_used"].astype(int)
    df_features["is_fraud"] = df_features["is_fraud"].astype(int)
    df_features["is_customer_details_changed_3_days_before_pickup"] = df_features[
        "is_customer_details_changed_3_days_before_pickup"
    ].astype(int)
    df_features["is_tracked_traffic_sessions"] = df_features[
        "is_tracked_traffic_sessions"
    ].astype(int)

    return df_features


# Check data has been updated in past 24 hours
def is_data_refreshed(config: dict) -> bool:
    """
    Checks the metadata of dm_rent.rent_bookings to verify that booking data has been refreshed in the past day.

    __PARAMETERS__
    config: dict
        Configuration dictionary containing BigQuery database and table to check

    __RETURNS__
    bool
        True if the latest update is recent (yesterday or today), otherwise raises StaleDataError
    """

    # Get date of yesterday
    yesterday = date.today() - timedelta(days=1)

    # Get database and table to check for refresh
    database = config["refresh_check"]["database"]
    table = config["refresh_check"]["table"]

    # Get table creation date from database metadata
    data_loader = DataLoader()
    features_query = f"SELECT creation_time FROM `sf-da-dwh.{database}.INFORMATION_SCHEMA.TABLES` WHERE table_name = '{table}'"
    creation_time = data_loader.run_query(features_query).iloc[0]["creation_time"]
    creation_date = creation_time.date()

    # Check table was updated since yesterday
    if creation_date < yesterday:
        raise StaleDataError(
            f"dm_rent.rent_bookings was not updated since before yesterday. Last update was on {creation_date}"
        )
    return True


def generate_missing_binarized_feature_columns(
    df_features: pd.DataFrame, top_binarized_multi_category_features: list
) -> pd.DataFrame:
    """
    Check that all provided columns are present in the DataFrame. If not generate the missing columns and set all their values to zero.

    __PARAMETERS__
    df_features: pd.DataFrame
        The input DataFrame containing booking-level features
    top_binarized_multi_category_features: list
        list of top binarized multi categorical features calculated during the training pipeline

    __RETURNS__
    pd.DataFrame
        The input DataFrame with any missing binarized feature columns added
    """
    # check for missing multicategorical features
    for binarized_feature in top_binarized_multi_category_features:
        # If column is missing generate it and set all values to 0
        if binarized_feature not in df_features.columns:
            df_features[binarized_feature] = 0
    return df_features
