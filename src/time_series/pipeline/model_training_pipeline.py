from src.time_series.constants import *
from src.time_series.entity.config_entity import ModelTrainerConfig
from src.time_series.config.configuration import ConfigurationManager
from src.time_series.components.model_training import ModelTrainer

STAGE_NAME="Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def InitializeModelTrainer(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train_model()

if __name__ == "__main__":
    try:
        model_trainer_pipeline = ModelTrainingPipeline()
        model_trainer_pipeline.InitializeModelTrainer()
    except Exception as e:
        raise e