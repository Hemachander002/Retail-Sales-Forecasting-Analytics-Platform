from src.time_series.entity.config_entity import (DataIngestionConfig,DataTransformationConfig,DataValidationConfig,ModelEvaluationConfig,ModelTrainerConfig)
from src.time_series.utils.common import read_yaml, create_directories,save_json, load_json
from src.time_series.constants import *

class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH,
                 schema_filepath = SCHEMA_FILE_PATH):
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)
        self.schema=read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config=DataIngestionConfig(
            root_dir= config.root_dir,
            main_source= config.main_source,
            holiday_source= config.holiday_source,
            stores_source= config.stores_source,
            transaction_source= config.transaction_source,
            oil_source= config.oil_source,
            train_source= config.train_source,
            local_data_file= config.local_data_file)
        
        return data_ingestion_config
    
    def get_data_validation_config(self)-> DataValidationConfig:
        config=self.config.data_validation
        schema = self.schema.COLUMNS
        create_directories([config.root_dir])

        data_validation_config=DataValidationConfig(
            root_dir=config.root_dir,
            unzip_data_dir=config.unzip_data_dir,
            STATUS_FILE=config.STATUS_FILE,
            all_schema= schema
        )
        return data_validation_config
    
    def get_data_transformation_config(self)-> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            holiday_source= config.holiday_source,
            oil_source= config.oil_source,
            transaction_source= config.transaction_source,
            stores_source= config.stores_source
        )
        return data_transformation_config
    
    def get_model_trainer_config(self)-> ModelTrainerConfig:
        config = self.config.model_trainer
        schema = self.schema.TARGET
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            valid_data_path=config.valid_data_path,
            label_encoder_path= config.label_encoder_path,
            model_name=config.model_name,
            target_column= schema.name
        )
        return model_trainer_config
    
    def get_model_evaluation_config(self)-> ModelEvaluationConfig:
        config = self.config.model_evaluation
        schema = self.schema.TARGET
        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            metric_file_name=config.metric_file_name,
            target_column= schema.name,
            mlflow_uri= 
        )
        return model_evaluation_config