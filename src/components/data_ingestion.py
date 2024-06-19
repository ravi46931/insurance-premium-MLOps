import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.constants import DATA_PATH
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig) -> None:
        self.data_ingestion_config = data_ingestion_config

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Data ingestion started..")
            df = pd.read_csv(DATA_PATH)
            os.makedirs(
                self.data_ingestion_config.DATA_INGESTION_ARTIFACT_DIR, exist_ok=True
            )
            df.to_csv(self.data_ingestion_config.DATA_FILE_PATH)
            data_ingestion_artifact = DataIngestionArtifact(
                self.data_ingestion_config.DATA_FILE_PATH
            )

            logging.info("Data ingestion completed..")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    dataingestion = DataIngestion(DataIngestionConfig())
    data_ingestion_artifact = dataingestion.initiate_data_ingestion()
    print(data_ingestion_artifact)
