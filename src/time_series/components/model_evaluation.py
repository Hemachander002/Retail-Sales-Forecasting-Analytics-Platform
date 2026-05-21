import os
import joblib
import dagshub
import mlflow
from mlflow.metrics import mae
import mlflow.sklearn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from urllib.parse import urlparse
from src.time_series.entity.config_entity import ModelEvaluationConfig
from src.time_series.utils.common import save_json
from src.time_series.config.configuration import ConfigurationManager
from pathlib import Path
import numpy as np
import pandas as pd


os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Hemachander002/Store-Sales---Time-Series-Forecasting.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "Hemachander002"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "254edea09d48445e8e8de0075ef293dfc5f9e30a"


class ModelEvaluator:
    def __init__(self,config: ModelEvaluationConfig):
        self.config = config
        self.model_path = config.model_path
        self.X_test = config.test_data_path
        self.y_test = config.target_column
        self.registry_uri = config.mlflow_uri
        self.metric_file_name = config.metric_file_name

    def eval_metrics(self,actual, predicted):
        r2 = r2_score(actual, predicted)
        mean_abs = mean_absolute_error(actual, predicted)

        return r2, mean_abs

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.X_test)
        y_test = test_data[self.y_test]
        X_test = test_data.drop(columns=[self.y_test,"date","id"], axis=1)
        mlflow.set_registry_uri(self.registry_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        model = joblib.load(self.model_path)

        with mlflow.start_run():
            predicted = model.predict(X_test)
            r2, mean_abs = self.eval_metrics(y_test, predicted)
            scores = {"r2": r2, "mean_absolute_error": mean_abs}
            save_json(path= Path(self.metric_file_name), data=scores)
            mlflow.log_params(model.get_params())
            mlflow.log_artifact(self.model_path, artifact_path="model")
            mlflow.log_metrics(scores)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, name =  "model", registered_model_name="Best_Model")
            else:
                mlflow.sklearn.log_model(model, name =  "model")


if __name__ == "__main__":
    try:
        config = ConfigurationManager()
        model_eval_config = config.get_model_evaluation_config()
        model_eval = ModelEvaluator(config=model_eval_config)
        model_eval.log_into_mlflow()
    except Exception as e:
        raise e