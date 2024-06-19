import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion from train pipeline..")
            dataingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact = dataingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed in train pipeline..")
            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation from train pipeline..")
            data_transformation = DataTransformation(
                data_ingestion_artifact, self.data_transformation_config
            )
            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )
            logging.info("Data transformation completed in train pipeline..")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            logging.info("Starting model trainer from train pipeline..")
            model_trainer = ModelTrainer(
                data_transformation_artifact, self.model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Model trainer completed in train pipeline..")
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def start_model_evaluation(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            logging.info("Starting model evaluation from train pipeline..")
            model_evaluation = ModelEvaluation(
                data_transformation_artifact,
                model_trainer_artifact,
                self.model_evaluation_config,
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logging.info("Model evaluation completed in train pipeline..")
            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Running train pipeline...")
            data_ingestion_artifact = self.start_data_ingestion()
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact
            )
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact
            )
            model_evaluation_artifact = self.start_model_evaluation(
                data_transformation_artifact, model_trainer_artifact
            )

            logging.info("Pipeline executed successfully..")
        except Exception as e:
            raise CustomException(e, sys)
