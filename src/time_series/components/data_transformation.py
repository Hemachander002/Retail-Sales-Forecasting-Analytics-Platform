import os
import re
import joblib
from narwhals import col
import numpy as np
from pyparsing import col
from src.time_series import logger
from src.time_series.entity.config_entity import (DataTransformationConfig)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class DataTransformation:
    def __init__(self,config:DataTransformationConfig):
        self.config=config
        self.data = pd.read_csv(self.config.data_path)
        self.holiday = pd.read_csv(self.config.holiday_source)
        self.stores = pd.read_csv(self.config.stores_source)
        self.oil = pd.read_csv(self.config.oil_source)
        self.transaction = pd.read_csv(self.config.transaction_source)
    
    def preprocess(self,holidays_df,oil_df,stores_df,transactions_df):
        try:
            data = self.data.copy()
            for df in [transactions_df,stores_df,holidays_df,oil_df,data]:
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
            data = data.merge(transactions_df,on=["date", "store_nbr"],how="left")
            data = data.merge(stores_df,on="store_nbr",how="left")
            data = data.merge(holidays_df,on="date",how="left")
            data.rename(columns={"type_y" : "holiday_type"},inplace=True)
            data = data.merge(oil_df,on="date",how="left")
            data.rename(columns={"type_x" : "store_type"},inplace=True)
            data["is_holiday"] = (data["holiday_type"].notnull().astype(int))
            data["holiday_type"] = data["holiday_type"].fillna("No Holiday")
            data["dcoilwtico"] = (data["dcoilwtico"].ffill())
            data["dcoilwtico"] = (data["dcoilwtico"].bfill())
            data["transactions"] = (data["transactions"].fillna(0))
            data["year"] = data["date"].dt.year
            data["month"] = data["date"].dt.month
            data["day"] = data["date"].dt.day
            data["day_of_week"] = data["date"].dt.dayofweek
            data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
            data["quarter"] = data["date"].dt.quarter
            data["is_month_start"] = (data["date"].dt.is_month_start.astype(int))
            data["is_month_end"] = (data["date"].dt.is_month_end.astype(int))
            data.drop(columns=["locale","locale_name","description","transferred"],inplace=True)
            cat_cols = ["family","city","state","store_type","holiday_type"]
            encoders = {}
            for col in cat_cols:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                encoders[col] = le
            data = data.sort_values(by=["store_nbr", "family", "date"])
            data["sales_lag_30"] = (data.groupby(["store_nbr", "family"])["sales"].shift(30))
            data["rolling_mean_7"] = (data.groupby(["store_nbr", "family"])["sales"].transform(lambda x:x.shift(1).rolling(7).mean()))
            data["rolling_std_30"] = (data.groupby(["store_nbr", "family"])["sales"].transform(lambda x:x.shift(1).rolling(30).std()))
            data.dropna(inplace=True)
            joblib.dump(encoders,os.path.join(self.config.root_dir,"label_encoder.joblib"))

            return data
        except Exception as e:
            logger.exception(e)
            raise e
        
    def real_transformation(self):

        holiday = self.holiday.copy()
        stores = self.stores.copy()
        oil = self.oil.copy() 
        transaction = self.transaction.copy() 
        real_data = self.preprocess(holidays_df=holiday,stores_df= stores,oil_df= oil , transactions_df= transaction)
        
        return real_data


    def split_data(self,df:pd.DataFrame):

        split_date = "2017-01-01"

        train_data = df[df["date"] < split_date]

        valid_data = df[df["date"] >= split_date]

        train_data.to_csv(os.path.join(self.config.root_dir,"training.csv"), index = False)
        valid_data.to_csv(os.path.join(self.config.root_dir,"valid.csv"), index = False)
