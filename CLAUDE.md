# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Poetry-managed Python 3.11 project. All targets are wrapped in the [Makefile](Makefile):

- `make install` — `poetry install` (also installs dev tools)
- `make format` — Black on `ds_fraud_detection_app` and `tests`
- `make lint` — Flake8 (config in [.flake8](.flake8); do not change, shared across DS projects)
- `make test` — Pytest against `tests/`
- `make all` — install + format + lint + test (this is what CI runs)
- `make build` — `poetry build` for package publish

Run a single test: `poetry run pytest tests/test_components/test_preprocess.py::test_name -v`

Run pipelines locally (require GCP auth):
- Train: `poetry run python -m ds_fraud_detection_app.run_train_pipeline --env dev [--n_eval_folds N]`
- Predict: `poetry run python -m ds_fraud_detection_app.run_predict_pipeline --env dev`

`--env` is the key knob: `dev` prefixes BigQuery output tables with `dev_` and targets the dev Vertex AI model registry entry; `main` writes to production tables/registry. Use `dev` for anything non-production.

## Auth for the private package source

`ds-package-classifier` is hosted in a private GCP Artifact Registry (`ds-packages` source in [pyproject.toml](pyproject.toml)). Before `poetry install` works locally:

```
poetry self add keyring keyrings.google-artifactregistry-auth
gcloud auth application-default login
```

CI uses `GCP_SA_KEY` and the same keyring plugin (see [.github/workflows/CI.yaml](.github/workflows/CI.yaml)).

## Architecture

This app trains an XGBoost fraud classifier for bookings and runs batch predictions over future-pickup bookings. Both pipelines are triggered from Airflow (in the `data-workflows` repo) — there is no online serving / API. The Docker image built by [CD_Docker.yaml](.github/workflows/CD_Docker.yaml) is a generic base image that Vertex AI Model Registry requires when hosting the model artifact; it is not used to serve requests from this app's pipelines. Two pipelines share a base class:

```
run_train_pipeline.py        run_predict_pipeline.py
        │                              │
        ▼                              ▼
TrainingPipeline ─────► BasePipeline ◄───── PredictPipeline
        │                                       │
        └──► components/ (data, preprocess, engineer_features, train, evaluate)
                          │
                          ├──► FraudClassifier (extends ds_package_classifier.MLClassifier)
                          ├──► utils/loaders/bigquery.py    (DataLoader)
                          └──► utils/loaders/vertex_ai.py   (upload_model / load_model_artifact)
```

Key conventions:

- **Config-driven**, not code-driven. [ds_fraud_detection_app/config/config.yaml](ds_fraud_detection_app/config/config.yaml) defines the feature lists (categorical / binary / numeric / multi-categorical), BigQuery source + destination tables, model hyperparameters, warning thresholds, and the GCP model-registry resource IDs per env. New features go in config first; the pipeline reads them generically. Load via `utils.utils.load_config("config.yaml")` (uses `importlib.resources`, so the YAML must be inside the package).
- **FraudClassifier** ([fraud_classifier.py](ds_fraud_detection_app/fraud_classifier.py)) is a thin subclass of `ds_package_classifier.MLClassifier`. It stores model-time state that must survive serialization to Vertex AI: `top_binarized_multi_category_features`, `medium_warning_level_threshold`, `high_warning_level_threshold`. The trained classifier is pickled with joblib and uploaded to GCS, then registered as a new version under the env-specific parent model in Vertex AI Model Registry.
- **Predict pipeline reconstructs the training feature space.** Multi-categorical columns (e.g. `traffic_sessions_sub_continents`) are binarized at train time; at predict time `generate_missing_binarized_feature_columns` fills in any columns the current batch is missing with zeros, using the list stored on the loaded `FraudClassifier`. Do not change the feature schema without retraining.
- **Warning levels** are derived from two thresholds on `predicted_fraud_probability`: medium (fixed 0.5 from config) and high (chosen at train time to maximize F-beta with `beta=5`, i.e. recall-heavy). Both are persisted on the classifier object.
- **BigQuery I/O** goes through `utils/loaders/bigquery.py::DataLoader` (project `sf-da-dwh`, location `europe-west3`). `append_table` has built-in exponential backoff on `rateLimitExceeded` — use it instead of writing raw `client.load_table_from_dataframe` calls.
- **Data freshness gate**: predict pipeline calls `is_data_refreshed(config)` and raises `StaleDataError` if `dm_rent.rent_bookings` hasn't been updated since yesterday. This is a hard stop, not a warning.

## Branching and release flow

- All work happens on feature branches off `dev`. CI ([CI.yaml](.github/workflows/CI.yaml)) runs `make all` on PRs into `dev` and fails if `git diff` is non-empty after formatting (i.e. run `make format` locally before pushing).
- `protect_main.yaml` blocks any PR into `main` that doesn't come from `dev`.
- Push to `main` triggers **two** workflows: [CD_Docker.yaml](.github/workflows/CD_Docker.yaml) builds and pushes the generic Vertex-AI hosting base image to `europe-west3-docker.pkg.dev/.../ds-images/ds-classifier-base:latest`, and [CD_Package.yaml](.github/workflows/CD_Package.yaml) builds the Poetry package, uploads it to the private `ds-packages` registry, then dispatches `install-airflow-packages.yml` in `roadsurfer-com/data-workflows` so Airflow picks up the new version. Bump `version` in [pyproject.toml](pyproject.toml) before merging to main. The package version, not the Docker image, is what actually changes the pipeline behavior in Airflow.
- Push to `dev` also rebuilds the base image (overwriting `:latest`) — be aware that the image tag is shared with main.
