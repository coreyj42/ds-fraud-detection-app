import argparse

from ds_fraud_detection_app.pipelines.train_pipeline import TrainingPipeline

# Get the parameters from CLI
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="dev")
parser.add_argument("--n_eval_folds", type=int, default=None)
args = parser.parse_args()

if __name__ == "__main__":
    print(args)
    # Run the train pipeline with the parameters from CLI
    train_pipeline = TrainingPipeline(env=args.env, n_eval_folds=args.n_eval_folds)
    train_pipeline.run_pipeline()
