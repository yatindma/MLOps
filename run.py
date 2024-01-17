import os

import click
from zenml.client import Client
from zenml.logger import get_logger

from pipelines import feature_engineering, training, inference

logger = get_logger(__name__)


@click.command()
@click.option(
    "--feature-pipeline",
    is_flag=True,
    help="Run the feature engineering pipeline.",
)
@click.option(
    "--training-pipeline",
    is_flag=True,
    help="Run the training pipeline.",
)
@click.option(
    "--inference-pipeline",
    is_flag=True,
    help="Run the inference pipeline.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable caching.",
)
def main(
        feature_pipeline: bool = False,
        training_pipeline: bool = False,
        inference_pipeline: bool = False,
        no_cache: bool = False,
):
    """
    Main entry point for the ZenML pipeline execution.

    This function orchestrates the execution of different pipelines (feature engineering,
    training, and inference) based on the provided command-line options.

    Parameters:
    feature_pipeline (bool): Flag to run the feature engineering pipeline.
    training_pipeline (bool): Flag to run the training pipeline.
    inference_pipeline (bool): Flag to run the inference pipeline.
    no_cache (bool): Flag to disable caching for pipeline runs.
    """
    client = Client()

    try:
        # Configure the path for pipeline configurations
        config_folder = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "configs",
        )

        # Feature Engineering Pipeline Execution
        if feature_pipeline:
            pipeline_args = {"enable_cache": not no_cache}
            pipeline_args["config_path"] = os.path.join(config_folder, "feature_engineering.yaml")
            feature_engineering.with_options(**pipeline_args)()
            logger.info("Feature Engineering pipeline finished successfully!")

            # Retrieve and log information using ZenML client
            train_dataset_artifact = client.get_artifact_version("dataset_trn")
            test_dataset_artifact = client.get_artifact_version("dataset_tst")
            logger.info(
                f"Feature Engineering Artifacts:\n"
                f"Train Dataset - Name: dataset_trn, Version: {train_dataset_artifact.version}\n"
                f"Test Dataset - Name: dataset_tst, Version: {test_dataset_artifact.version}"
            )

        # Training Pipeline Execution
        if training_pipeline:
            pipeline_args = {"enable_cache": not no_cache}
            pipeline_args["config_path"] = os.path.join(config_folder, "training_rf_regressor.yaml")
            training_run = training.with_options(**pipeline_args)()
            logger.info("Training pipeline finished successfully!")

        # Inference Pipeline Executionz
        if inference_pipeline:
            pipeline_args = {"enable_cache": not no_cache}
            pipeline_args["config_path"] = os.path.join(config_folder, "inference.yaml")
            inference.with_options(**pipeline_args)()
            logger.info("Inference pipeline finished successfully!")

    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}")
        raise


if __name__ == "__main__":
    main()
