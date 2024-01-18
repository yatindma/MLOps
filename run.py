import os
import click
from zenml.client import Client
from zenml.logger import get_logger
from pipelines import feature_engineering, training, inference

logger = get_logger(__name__)

@click.command()
@click.option("--feature-pipeline", is_flag=True, help="Run the feature engineering pipeline.")
@click.option("--training-pipeline", is_flag=True, help="Run the training pipeline.")
@click.option("--inference-pipeline", is_flag=True, help="Run the inference pipeline.")
@click.option("--no-cache", is_flag=True, help="Disable caching.")
def main(feature_pipeline: bool = False, training_pipeline: bool = False, inference_pipeline: bool = False, no_cache: bool = False):
    """
    Main entry point for the ZenML pipeline execution.
    Orchestrates the execution of different pipelines based on command-line options.
    """
    client = Client()
    config_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")

    try:
        if feature_pipeline:
            logger.info("Running the feature engineering pipeline.")
            pipeline_args = {"enable_cache": not no_cache, "config_path": os.path.join(config_folder, "feature_engineering.yaml")}
            feature_engineering.with_options(**pipeline_args)()
            logger.info("Feature engineering pipeline completed successfully.")
            log_artifact_info(client, "dataset_trn", "dataset_tst")

        if training_pipeline:
            logger.info("Running the training pipeline.")
            pipeline_args = {"enable_cache": not no_cache, "config_path": os.path.join(config_folder, "training_rf_regressor.yaml")}
            training.with_options(**pipeline_args)()
            logger.info("Training pipeline completed successfully.")

        if inference_pipeline:
            logger.info("Running the inference pipeline.")
            pipeline_args = {"enable_cache": not no_cache, "config_path": os.path.join(config_folder, "inference.yaml")}
            inference.with_options(**pipeline_args)()
            logger.info("Inference pipeline completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
        raise

def log_artifact_info(client, *artifact_names):
    for name in artifact_names:
        artifact = client.get_artifact_version(name)
        logger.info(f"Artifact: {name}, Version: {artifact.version}")

if __name__ == "__main__":
    main()
