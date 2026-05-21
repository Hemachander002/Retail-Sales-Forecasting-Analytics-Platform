import os
import urllib.request as request
from src.time_series import logger
import zipfile
from src.time_series.entity.config_entity import (DataIngestionConfig)
import pandas as pd
from pathlib import Path

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config
    

    def data_extraction(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        main = pd.read_csv(self.config.main_source)
        oil = pd.read_csv(self.config.oil_source)
        holiday = pd.read_csv(self.config.holiday_source)
        stores = pd.read_csv(self.config.oil_source)
        transaction = pd.read_csv(self.config.transaction_source)
        train = pd.read_csv(self.config.train_source)

        main.to_csv(os.path.join(self.config.root_dir,"test.csv"),index = False)
        oil.to_csv(os.path.join(self.config.root_dir,"oil.csv"),index = False)
        holiday.to_csv(os.path.join(self.config.root_dir,"holiday.csv"),index = False)
        stores.to_csv(os.path.join(self.config.root_dir,"stores.csv"),index = False)
        transaction.to_csv(os.path.join(self.config.root_dir,"transaction.csv"),index = False)
        train.to_csv(os.path.join(self.config.root_dir,"train.csv"),index = False)
        