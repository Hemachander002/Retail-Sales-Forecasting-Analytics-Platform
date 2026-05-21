from src.time_series.constants import *
from src.time_series.entity.config_entity import ModelTrainerConfig
from src.time_series.config.configuration import ConfigurationManager
import os
from src.time_series.components.model_evaluation import ModelEvaluator
import joblib

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def InitiateModelEvaluation(self):
        config = ConfigurationManager()
        model_eval_config = config.get_model_evaluation_config()
        model_eval = ModelEvaluator(config=model_eval_config)
        model_eval.log_into_mlflow()

if __name__ == "__main__":
    try:
        model_eval_pipeline = ModelEvaluationPipeline()
        model_eval_pipeline.InitiateModelEvaluation()
    except Exception as e:
        raise e