import argparse

from ds_fraud_detection_app.pipelines.predict_pipeline import PredictPipeline

# Get the parameters from CLI
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="dev")
parser.add_argument("--n_eval_folds", type=int, default=None)
args = parser.parse_args()

if __name__ == "__main__":
    print(args)
    # Run the train pipeline with the parameters from CLI
    predict_pipeline = PredictPipeline(env=args.env)
    predict_pipeline.run_pipeline()
