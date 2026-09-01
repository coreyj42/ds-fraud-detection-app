from typing import Optional

from ds_package_classifier.utils.utils import calculate_imbalance_ratio

from ds_fraud_detection_app.pipelines.base import BasePipeline
from ds_fraud_detection_app.components.data import load_bigquery_features
from ds_fraud_detection_app.components.preprocess import (
    filter_rows,
    preprocess_training_features,
    convert_data_types,
)
from ds_fraud_detection_app.components.engineer_features import engineer_all_features
from ds_fraud_detection_app.components.train import (
    get_top_feature_categories,
    train_final_model,
    calculate_high_warning_threshold,
)
from ds_fraud_detection_app.components.evaluate import (
    calculate_trained_classifier_metrics,
    log_threshold_metrics,
)

from ds_fraud_detection_app.components.data import (
    save_classifier_metrics,
    upload_model_to_vertex_ai,
    create_all_bookings_fraud_predictions_table,
    append_predictions_table,
)


class TrainingPipeline(BasePipeline):

    def __init__(
        self,
        env: str = "dev",
        pipeline_name="Training Pipeline",
        n_eval_folds: Optional[int] = None,
    ):
        """
        A pipeline class for training a fraud detection model using booking-level features.

        This pipeline performs the following steps:

        Step 1: Load data from BigQuery
            - Step 1.1: Load wrangled booking-level features from the specified BigQuery table.

        Step 2: Prepare training data
            - Step 2.1: Convert defined features to correct data types.
            - Step 2.2: Filter data to remove outliers and invalid bookings.
            - Step 2.3: Determine the number of evaluation folds if not manually provided.
            - Step 2.4: Calculate class imbalance ratio for weighting during training.
            - Step 2.5: Engineer additional features to enhance model performance.
            - Step 2.6: Preprocess features (e.g., handle missing values, binarize multi-categorical features).
            - Step 2.7: Record expanded multi-categorical feature names for later use.

        Step 3: Train the model
            - Step 3.1: Train an initial model to identify top-performing features.
            - Step 3.2: Train the final model using selected features and class weights.
            - Step 3.3: Calculate the high warning level threshold for fraud detection.

        Step 4: Evaluate the model
            - Step 4.1: Compute cross-validation metrics and log them.
            - Step 4.2: Calculate cross-validation fraud probabilities.
            - Step 4.3: Build a results table with predictions and warning levels.

        Step 5: Save the model and results (only in 'main' environment)
            - Step 5.1: Save evaluation metrics to BigQuery.
            - Step 5.2: Save cross-validation predictions to BigQuery.
            - Step 5.3: Upload the trained model to Vertex AI.

        Final Step: Complete the training pipeline

        __PARAMETERS__
        env: str
            The environment in which the pipeline is running (e.g., 'dev', 'main').
        pipeline_name: str
            A descriptive name for the pipeline instance.
        n_eval_folds: Optional[int]
            Number of evaluation folds for cross-validation; if not provided, it is inferred from the data.

        __ATTRIBUTES__
        top_categories: list
            Most important features selected during initial model training.
        fraud_classifier: FraudClassifier
            The trained fraud classifier model.
        imbalance_ratio: float
            Ratio of class imbalance used for weighting during training.
        top_binarized_multi_category_features: list
            Selected top multi-categorical features after binarization.
        cross_val_metrics: pd.DataFrame
            Cross-validation metrics for model evaluation.
        threshold_cross_val_metrics: pd.DataFrame
            Threshold-based evaluation metrics.
        datetime_tag: str
            Timestamp used to tag saved artifacts and metrics.
        database: str
            Name of the BigQuery database from which features are loaded.
        features_table: str
            Name of the BigQuery table containing the booking-level features.
        target_column: str
            Name of the target column used for fraud classification.
        categorical_features: list[str]
            List of categorical feature column names used in training.
        binary_features: list[str]
            List of binary feature column names used in training.
        numeric_features: list[str]
            List of numeric feature column names used in training.
        multi_categorical_features: list[str]
            List of multi-categorical feature column names to be binarized.
        is_n_eval_folds_provided: bool
            Indicates whether the number of evaluation folds was manually specified.
        """

        super().__init__(env, pipeline_name)
        self.n_eval_folds = n_eval_folds
        self.top_categories = None
        self.fraud_classifier = None
        self.imbalance_ratio = (
            None  # Used to weigh classes during training to handle class imbalance
        )
        self.top_binarized_multi_category_features = (
            None  # Stores top multi-categorical feature categories
        )
        self.cross_val_metrics = None
        self.threshold_cross_val_metrics = None

        # set constants from config.yaml
        self.database = self.config["data"]["database"]
        self.features_table = self.config["data"]["features_table"]
        self.target_column = self.config["data"]["target_column"]
        self.categorical_features = self.config["features"]["categorical_features"]
        self.binary_features = self.config["features"]["binary_features"]
        self.numeric_features = self.config["features"]["numeric_features"]
        self.multi_categorical_features = self.config["features"][
            "multi_categorical_features"
        ]

        # Create boolean variable to record if n_eval_folds was manually set
        if self.n_eval_folds:
            self.is_n_eval_folds_provided = True
        else:
            self.is_n_eval_folds_provided = False

    def run_pipeline(self):
        # Step 1: load data from BigQuery
        self.logger.info(
            f"Step 1: Load the data from BigQuery: {self.database}.{self.features_table}"
        )

        # Step 1.1 Load the data features from BigQuery
        self.logger.info(
            f"Step 1.1: Load wrangled booking level features: {self.database}.{self.features_table}"
        )
        df_features = load_bigquery_features(self.database, self.features_table)

        # Step 2: Prepare training data
        self.logger.info("Step 2: Prepare training data")

        # Step 2.1 Convert defined features to correct data types
        self.logger.info("Step 2.1 Convert defined features to correct data types")
        df_features = convert_data_types(df_features)

        # Step 2.2: Filter training data to remove outliers and other undesirable bookings
        self.logger.info(
            "Step 2.2: Filter training data to remove outliers and other undesirable bookings"
        )
        df_features = filter_rows(df_features, self.config)

        # Step 2.3: Calculate n_eval_folds if it was not manually set
        self.logger.info("Step 2.3: Calculate n_eval_folds if it was not manually set")
        if not self.is_n_eval_folds_provided:
            self.n_eval_folds = len(df_features[df_features[self.target_column] == 1])

        # Step 2.4: Calculate imbalance ratio for class weight scaling
        self.logger.info("Step 2.4: Calculate imbalance ratio for class weight scaling")
        self.imbalance_ratio = calculate_imbalance_ratio(
            df_features, self.target_column
        )

        # Step 2.5: Engineer further features
        self.logger.info("Step 2.5: Engineer further features")
        df_features = engineer_all_features(df_features)

        # Step 2.6: Apply final preprocessing to features
        self.logger.info("Step 2.6: Apply final preprocessing to training features")
        df_features, expanded_multi_categorical_features = preprocess_training_features(
            df_features, self.config
        )

        # Step 2.7: Record expanded multi categorical features
        self.logger.info("Step 2.7: Record expanded multi categorical features")
        self.expanded_multi_categorical_features = expanded_multi_categorical_features

        # Step 3: Train the model
        self.logger.info("Step 3: Train the model")

        # Step 3.1: Train initial model and extract most used features
        self.logger.info("Step 3.1: Train initial model and extract most used features")
        self.top_categories, self.top_binarized_multi_category_features = (
            get_top_feature_categories(
                df_features,
                self.config,
                self.expanded_multi_categorical_features,
                self.imbalance_ratio,
            )
        )

        # Step 3.2: Train the final model
        self.logger.info("Step 3.2: Train the final model")
        self.fraud_classifier = train_final_model(
            df_features,
            self.config,
            self.top_categories,
            self.top_binarized_multi_category_features,
            self.imbalance_ratio,
        )

        # Step 3.3: Extract the thresholds for warning flag levels
        self.logger.info("Step 3.3: Extract the thresholds for warning flag levels")
        self.fraud_classifier.high_warning_level_threshold = (
            calculate_high_warning_threshold(
                self.fraud_classifier, self.config, self.n_eval_folds
            )
        )

        # Step 4: Model Evaluation
        self.logger.info("Step 4: Model Evaluation")

        # Step 4.1: Calculate model cross-validation metrics
        self.logger.info("Step 4.1: Calculate model cross-validation metrics")
        df_threshold_metrics = calculate_trained_classifier_metrics(
            self.fraud_classifier, self.n_eval_folds
        )
        log_threshold_metrics(df_threshold_metrics, self.logger)

        # Step 4.2: Calculate cross-val fraud probabilities
        self.logger.info("Step 4.2: Calculate cross-val fraud probabilities")
        y_pred_cross_val_proba = self.fraud_classifier.cross_val_predict_proba(
            n_splits=self.n_eval_folds
        )[:, 1]

        # Step 4.3: Build cross-val prediction results table
        self.logger.info("Step 4.3: Build cross-val prediction results table")
        df_results = create_all_bookings_fraud_predictions_table(
            df_features,
            self.config,
            self.fraud_classifier,
            y_pred_cross_val_proba,
            include_target=True,
        )

        # Step 5: Save model and results
        self.logger.info("Step 5: Save model and results")

        # Step 5.1 Save metrics to BigQuery
        self.logger.info("Step 5.1: Save metrics to BigQuery")
        save_classifier_metrics(
            df_threshold_metrics,
            self.fraud_classifier,
            self.datetime_tag,
            self.n_eval_folds,
            self.env,
        )

        # Step 5.2: Save training data cross-validation predictions to BigQuery
        self.logger.info(
            "Step 5.2: Save training data cross-validation predictions to BigQuery"
        )
        database_name = self.config["train_cross_val_predictions"]["database"]
        table_name = self.config["train_cross_val_predictions"]["table"]
        append_predictions_table(
            df_results, self.logger, database_name, table_name, self.env
        )

        # Step 5.3 Upload model to Vertex AI
        self.logger.info("Step 5.3: Upload model to Vertex AI")
        upload_model_to_vertex_ai(
            self.fraud_classifier, self.datetime_tag, config=self.config, env=self.env
        )

        # Pipeline finished
        self.logger.info("Training pipeline complete")
