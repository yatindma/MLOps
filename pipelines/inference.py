from zenml import get_pipeline_context, pipeline
from zenml.logger import get_logger
from steps import data_loader, inference_predict, inference_preprocessor

logger = get_logger(__name__)

@pipeline
def inference(random_state: str, target: str):
    """
    Model inference pipeline.
    """
    try:
        logger.info("Starting inference pipeline.")
        model = get_pipeline_context().model_version.get_artifact("trained_model")
        preprocess_pipeline = get_pipeline_context().model_version.get_artifact("preprocess_pipeline")
        df_inference = data_loader(random_state=random_state, is_inference=True)
        df_inference = inference_preprocessor(dataset_inf=df_inference, preprocess_pipeline=preprocess_pipeline, target=target)
        inference_predict(model=model, dataset_inf=df_inference)
        logger.info("Inference pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Error in inference pipeline: {e}", exc_info=True)
        raise
