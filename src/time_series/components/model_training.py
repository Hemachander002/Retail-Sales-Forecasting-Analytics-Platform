from re import search

from src.time_series.constants import *
from src.time_series.entity.config_entity import ModelTrainerConfig
from src.time_series.config.configuration import ConfigurationManager
from sklearn.model_selection import RandomizedSearchCV,TimeSeriesSplit
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error
import pandas as pd


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def train_model(self):
        train_data = pd.read_csv(self.config.train_data_path)
        valid_data = pd.read_csv(self.config.valid_data_path)
        x_train = train_data.drop(columns=[self.config.target_column,"date","id"], axis=1)
        y_train = train_data[self.config.target_column]
        x_valid = valid_data.drop(columns=[self.config.target_column,"date","id"], axis=1)
        y_valid = valid_data[self.config.target_column]
        xgb_model = XGBRegressor(n_estimators=200,
                                 learning_rate=0.1,
                                 max_depth=6,
                                 subsample=0.8,
                                 colsample_bytree=0.8,
                                 random_state=42)
        params = {"n_estimators": [100, 200, 300],
                  "learning_rate": [0.01, 0.05, 0.1],
                  "max_depth": [4, 6, 8],
                  "subsample": [0.7, 0.8, 1.0],
                  "colsample_bytree": [0.7, 0.8, 1.0]}
        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(estimator=xgb_model,param_distributions=params,
                                    n_iter=10,scoring="neg_root_mean_squared_error",
                                    cv=tscv,verbose=1,random_state=42,n_jobs=-1)

        search.fit(x_train, y_train)
        best_model = search.best_estimator_
        joblib.dump(best_model, os.path.join(self.config.root_dir, self.config.model_name))

        return best_model

