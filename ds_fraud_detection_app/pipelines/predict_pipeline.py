from datetime import date
from copy import deepcopy

from ds_fraud_detection_app.pipelines.base import BasePipeline
from ds_fraud_detection_app.components.data import (
    load_bigquery_features,
    download_model,
    create_all_bookings_fraud_predictions_table,
    append_predictions_table,
    create_warnings_table,
)
from ds_fraud_detection_app.components.preprocess import (
    preprocess_training_features,
    convert_data_types,
    is_data_refreshed,
    generate_missing_binarized_feature_columns,
)
from ds_fraud_detection_app.components.engineer_features import engineer_all_features


class PredictPipeline(BasePipeline):

    def __init__(
        self,
        env: str = "dev",
        pipeline_name="Predict Pipeline",
    ):
        super().__init__(env, pipeline_name)
        self.env = env

        # set constants from config.yaml
        self.database = self.config["data"]["database"]
        self.features_table = self.config["data"]["features_table"]

    def run_pipeline(self):

        # Step 1: load data and artifacts from GCP
        self.logger.info(
            f"Step 1: Load the data from BigQuery: {self.database}.{self.features_table}"
        )

        # Step 1.1: Load the data features from BigQuery
        self.logger.info(
            f"Step 1.1: Load wrangled booking level features: {self.database}.{self.features_table}"
        )
        df_features = load_bigquery_features(self.database, self.features_table)

        # Step 1.2: Load model artifact from GCP
        self.logger.info("Step 1.2: Load model artifact from GCP")
        fraud_classifier, version_tag = download_model(self.config, self.env)

        # Step 2: Prepare prediction data
        self.logger.info("Step 2: Prepare training data")

        # Step 2.1: Convert defined datetime features to dates as needed
        self.logger.info(
            "Step 2.1: Convert defined datetime features to dates as needed"
        )
        df_features = convert_data_types(df_features)

        # Step 2.2 Check data has been updated in past 24 hours
        self.logger.info("Step 2.2 Check data has been updated in past 24 hours")
        is_data_refreshed(self.config)

        # Step 2.3: Filter bookings with pickup_dates from today onwards
        self.logger.info(
            "Step 2.3: Filter bookings with pickup_dates from today onwards"
        )
        df_features = deepcopy(df_features[df_features["pickup_date"] >= date.today()])
        df_features.reset_index(inplace=True, drop=True)

        # Step 2.4: Engineer further features
        self.logger.info("Step 2.5: Engineer further features")
        df_features = engineer_all_features(df_features)

        # Step 2.6: Apply final preprocessing to features
        self.logger.info("Step 2.6: Apply final preprocessing to training features")
        df_features, _ = preprocess_training_features(df_features, self.config)

        # Step 2.7: Check for and fill missing binarised features needed for predict
        self.logger.info(
            "Step: 2.7: Check for and fill missing binarised features needed for predict"
        )
        df_features = generate_missing_binarized_feature_columns(
            df_features, fraud_classifier.top_binarized_multi_category_features
        )

        # Step 3 Make predictions
        self.logger.info("Step 3 Make predictions")

        # Step 3.1: Predict fraud probabilities
        self.logger.info("Step 3.1: Predict fraud probabilities")
        y_pred_proba = fraud_classifier.predict_proba(df_features)[:, 1]

        # 3.2: Create prediction results table
        self.logger.info("Step 3.2: Create prediction results table")
        df_results = create_all_bookings_fraud_predictions_table(
            df_features, self.config, fraud_classifier, y_pred_proba
        )

        # 3.3: Create warning table containing only risky bookings
        self.logger.info("Step 3.3: Create warning table")
        df_warnings = create_warnings_table(df_results)

        # Step 4: Save predictions
        self.logger.info("Step 4: Save predictions")

        # 4.2: Save all predictions to table for current model version
        # Save to latest predictions
        self.logger.info(
            "Step 4.2: Save all predictions to table for current model version"
        )
        database_name = self.config["predictions"]["database"]
        table_name = self.config["predictions"]["all_bookings_predictions_table"]
        append_predictions_table(
            df_results,
            self.logger,
            database_name,
            table_name,
            self.env,
            table_tag=f"model_v_{version_tag}",
        )

        # Step 4.3: Save warnings of risky bookings
        self.logger.info("Step 4.3: Save warnings of risky bookings")
        database_name = self.config["predictions"]["database"]
        table_name = self.config["predictions"]["warnings_table"]
        append_predictions_table(
            df_warnings, self.logger, database_name, table_name, self.env
        )

        # Pipeline finished
        self.logger.info("Predict pipeline complete")
