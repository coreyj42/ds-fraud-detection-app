import re
import pandas as pd
import country_converter as coco
from copy import deepcopy
from typing import Union
import numpy as np


def engineer_is_online_bank_used(
    df_features: pd.DataFrame, online_banks: list[str]
) -> pd.DataFrame:
    """
    Build a boolean feature column to signal if any of the banks in bank_names are online only banks.

    __PARAMETERS__
    df_features: pd.DataFrame
        The input DataFrame

    __RETURNS__
    pd.DataFrame
        The original dataframe with a new column is_online_bank_used added
    """

    # Build a combined regex pattern from the list
    pattern = r"|".join([rf"\b{name}\b" for name in online_banks])

    # Create a new column with 1 if any match is found, else 0
    df_features["is_online_bank_used"] = df_features["bank_names"].apply(
        lambda lst: int(
            any(re.search(pattern, item.strip(), re.IGNORECASE) for item in lst)
        )
    )

    return df_features


def engineer_is_non_card_payment_method_used(
    df_features: pd.DataFrame, non_card_payment_methods: list[str]
) -> pd.DataFrame:
    """
    Build a boolean feature column to signal if any of the payment methods in payment_methods are non-card payment methods.

    __PARAMETERS__
    df_features: pd.DataFrame
        The input DataFrame

    __RETURNS__
    pd.DataFrame
        The original dataframe with a new column is_non_card_payment_method_used added
    """

    # Create a new column with 1 if any non-card payment method is found, else 0
    df_features["is_non_card_payment_method_used"] = df_features[
        "payment_methods"
    ].apply(lambda lst: int(any(item in non_card_payment_methods for item in lst)))

    return df_features


def _generate_country_to_code_mapping(
    country_names: Union[list[str], np.ndarray]
) -> dict:
    """
    Generate a dictionary mapping country names to their ISO2 country codes.

    __PARAMETERS__
    country_names: Union[list[str], np.ndarray]
        A list of country names to be converted

    __RETURNS__
    dict
        A dictionary where keys are country names and values are their corresponding ISO2 codes in lowercase
    """

    country_codes = coco.convert(names=country_names, to="ISO2", not_found=None)
    country_codes = [code.lower() for code in country_codes]
    country_mapping_dict = {}

    for i in range(len(country_names)):
        country_mapping_dict[country_names[i]] = country_codes[i]

    return country_mapping_dict


def _build_multcat_match_flag(
    df_features: pd.DataFrame, cat_column: str, multicat_col: str, new_col
) -> pd.DataFrame:
    """
    Build a boolean feature column to signal if the value in a single-category column is present in a multi-category column.

    __PARAMETERS__
    df_features: pd.DataFrame
        The input DataFrame
    cat_column: str
        The name of the column containing a single category value
    multicat_col: str
        The name of the column containing a list of multiple category values
    new_col: str
        The name of the new column to be added

    __RETURNS__
    pd.DataFrame
        The original dataframe with a new column added indicating whether the single category is present in the multi-category list
    """

    df_features[new_col] = df_features.apply(
        lambda row: int(row[cat_column] in row[multicat_col]), axis=1
    )
    return df_features


def build_residence_country_code(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Build a column containing ISO2 country codes mapped from the customer's residence country.

    __PARAMETERS__
    df_features: pd.DataFrame
        The input DataFrame

    __RETURNS__
    pd.DataFrame
        The original dataframe with a new column residence_country_code added
    """
    unique_residence_countries = df_features["residence_country"].unique()
    residence_country_code_mapping = _generate_country_to_code_mapping(
        unique_residence_countries
    )
    df_features["residence_country_code"] = df_features["residence_country"].map(
        residence_country_code_mapping
    )
    return df_features


def engineer_is_residence_bank_country_matched(
    df_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a boolean feature column to signal if the country of any banks used for payments match the customer's residence country.

    __PARAMETERS__
    df_features: pd.DataFrame
        The input DataFrame

    __RETURNS__
    pd.DataFrame
        The original dataframe with a new column is_residence_bank_country_matched added
    """
    tmp_df_features = deepcopy(df_features)

    # Match residence country code to bank country code
    tmp_df_features = _build_multcat_match_flag(
        tmp_df_features,
        "residence_country_code",
        "bank_country_codes",
        "is_residence_bank_country_matched",
    )

    # Add is_residence_bank_country_matched to original df_features
    df_features["is_residence_bank_country_matched"] = tmp_df_features[
        "is_residence_bank_country_matched"
    ]

    return df_features


def _get_unique_strings_from_list_column(
    df_features: pd.DataFrame, column_name: str
) -> list:
    """
    Extract all unique string values from a column containing lists of strings.

    __PARAMETERS__
    df: pd.DataFrame
        The input DataFrame
    column_name: str
        The name of the column containing lists of strings

    __RETURNS__
    list[str]
        A list of unique strings found across all lists in the specified column
    """
    return list(set(item for sublist in df_features[column_name] for item in sublist))


def _map_list_values(
    df_features: pd.DataFrame, mapping_dict: dict, column_to_map: str, new_column: str
) -> pd.DataFrame:
    """
    Map values in a list-column to new values using a provided dictionary.

    __PARAMETERS__
    df: pd.DataFrame
        The input DataFrame
    mapping_dict: dict
        A dictionary used to map original values to new values
    column_to_map: str
        The name of the column containing lists of values to be mapped
    new_column: str
        The name of the new column to store the mapped lists

    __RETURNS__
    pd.DataFrame
        The original dataframe with a new column containing mapped values
    """
    df_features[new_column] = df_features[column_to_map].apply(
        lambda country_list: [mapping_dict.get(country) for country in country_list]
    )
    return df_features


def _build_multi_cat_country_codes(
    df_features: pd.DataFrame, country_names_column: str, new_column: str
) -> pd.DataFrame:
    """
    Build a new column containing ISO2 country codes mapped from a list-column of country names.

    __PARAMETERS__
    df: pd.DataFrame
        The input DataFrame
    country_names_column: str
        The name of the column containing lists of country names
    new_col: str
        The name of the new column to store the mapped ISO2 country codes

    __RETURNS__
    pd.DataFrame
        The original dataframe with a new column containing ISO2 country codes
    """

    country_names = _get_unique_strings_from_list_column(
        df_features, country_names_column
    )
    country_mapping_dict = _generate_country_to_code_mapping(country_names)
    df_features = _map_list_values(
        df_features, country_mapping_dict, country_names_column, new_column
    )

    return df_features


def engineer_is_residence_traffic_sessions_country_matched(
    df_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    __PARAMETERS__
    df_features: pd.DataFrame
        The input DataFrame

    __RETURNS__
    pd.DataFrame
        The original dataframe with a new column is_residence_traffic_sessions_country_matched added
    """
    tmp_df_features = deepcopy(df_features)
    tmp_df_features = _build_multi_cat_country_codes(
        tmp_df_features, "traffic_sessions_countries", "traffic_sessions_country_codes"
    )

    tmp_df_features = _build_multcat_match_flag(
        tmp_df_features,
        "residence_country_code",
        "traffic_sessions_country_codes",
        "is_residence_traffic_sessions_country_matched",
    )
    df_features["is_residence_traffic_sessions_country_matched"] = tmp_df_features[
        "is_residence_traffic_sessions_country_matched"
    ]

    return df_features
