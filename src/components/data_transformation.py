import os
import sys
import joblib
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataTransformationConfig
from src.constants import PREPROCESSOR_FILE_NAME
from src.utils.utils import label_encode_column, encode_region_column, get_feature_names
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_config = data_transformation_config

    def transform_data(self, df):
        try:
            X = df.drop(columns="expenses", axis=1)
            y = df[["expenses"]]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_df = X_train.reset_index().drop("index", axis=1)
            X_test = X_test.reset_index().drop("index", axis=1)
            Y_df = y_train.reset_index().drop("index", axis=1)
            y_test = y_test.reset_index().drop("index", axis=1)

            """Encoding region column"""
            region_features = ["region"]
            region_transformer = Pipeline(
                steps=[
                    ("label", FunctionTransformer(encode_region_column, validate=False))
                ]
            )

            """Encoding sex, smoker, age, bmi columns """
            # Define preprocessing steps for each type of column
            categorical_features = ["sex", "smoker"]
            categorical_transformer = Pipeline(
                steps=[("onehot", OneHotEncoder(drop="first"))]
            )
            num_features = ["age", "bmi"]
            num_transformer = Pipeline(steps=[("scaler", StandardScaler())])

            # Create a ColumnTransformer to apply different preprocessing steps to different columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ("region", region_transformer, region_features),
                    ("cat", categorical_transformer, categorical_features),
                    ("num", num_transformer, num_features),
                ]
            )

            # Fit and transform the data
            train_transformed_data = preprocessor.fit_transform(
                X_df[["region", "sex", "smoker", "age", "bmi"]]
            )

            # Get the feature names after transformation
            feature_names = get_feature_names(
                preprocessor, X_df[["region", "sex", "smoker", "age", "bmi"]].columns
            )

            # Convert the transformed data to a DataFrame
            train_transformed_df = pd.DataFrame(
                train_transformed_data, columns=feature_names
            )
            train_df = pd.concat(
                [train_transformed_df, X_df[["children"]], Y_df[["expenses"]]], axis=1
            )

            test_transformed_data = preprocessor.transform(
                X_test[["region", "sex", "smoker", "age", "bmi"]]
            )
            test_transformed_df = pd.DataFrame(
                test_transformed_data, columns=feature_names
            )
            test_df = pd.concat(
                [test_transformed_df, X_test[["children"]], y_test[["expenses"]]],
                axis=1,
            )

            return train_df, test_df, preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data transformation initiated...")
            df = pd.read_csv(self.data_ingestion_artifact.data_file_path)
            train_df, test_df, preprocessor = self.transform_data(df)
            os.makedirs(
                self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACT_DIR,
                exist_ok=True,
            )
            # Saving transform dataframe
            train_df.to_csv(self.data_transformation_config.TRAIN_TRANSFORM_FILE_PATH)
            test_df.to_csv(self.data_transformation_config.TEST_TRANSFORM_FILE_PATH)
            # Saving preprocessor object
            joblib.dump(
                preprocessor, self.data_transformation_config.PREPROCESSOR_FILE_PATH
            )
            os.makedirs("models", exist_ok=True)
            preprocessor_path = "models/" + PREPROCESSOR_FILE_NAME
            joblib.dump(preprocessor, preprocessor_path)

            data_transformation_artifact = DataTransformationArtifact(
                self.data_transformation_config.TRAIN_TRANSFORM_FILE_PATH,
                self.data_transformation_config.TEST_TRANSFORM_FILE_PATH,
                self.data_transformation_config.PREPROCESSOR_FILE_PATH,
            )

            logging.info("Data Transformation completed...")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)





