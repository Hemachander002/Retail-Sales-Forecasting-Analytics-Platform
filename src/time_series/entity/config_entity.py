from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    main_source: Path
    train_source: Path
    holiday_source: Path
    stores_source: Path
    transaction_source: Path
    oil_source: Path
    local_data_file: Path

@dataclass
class DataValidationConfig:
    root_dir: Path
    unzip_data_dir: Path
    STATUS_FILE: str
    all_schema : dict

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path : Path
    holiday_source: Path
    stores_source: Path
    transaction_source: Path
    oil_source: Path

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path : Path
    valid_data_path : Path
    label_encoder_path : Path
    model_name : str
    target_column : str

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_name: Path
    target_column: str
    mlflow_uri: str