from src.time_series import logger
from src.time_series.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.time_series.pipeline.data_validation_pipeline import DataValidationPipeline
from src.time_series.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.time_series.pipeline.model_training_pipeline import ModelTrainingPipeline


STAGE_NAME="Data Ingestion Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj=DataIngestionTrainingPipeline()
    obj.InitiateDataIngestionTrainingPipeline()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="Data Validation Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj=DataValidationPipeline()
    obj.InitiateDataValidation()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME="Data Transformation Stage"
try:
    logger.info("Starting data transformation pipeline...")
    data_transformation_pipeline = DataTransformationPipeline()
    data_transformation_pipeline.InitializeDataTransformationPipeline()
    logger.info("Data transformation pipeline completed successfully!")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="Model Training Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    model_trainer_pipeline = ModelTrainingPipeline()
    model_trainer_pipeline.InitializeModelTrainer()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e




