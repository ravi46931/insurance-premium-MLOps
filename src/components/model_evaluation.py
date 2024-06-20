import os
import sys
import time
import json
import dagshub
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pickle
import pandas as pd
import warnings
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.constants.hyperparameter import *
from src.constants import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)
from src.entity.config_entity import ModelTrainerConfig, ModelEvaluationConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluation:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_config: ModelEvaluationConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_config = model_evaluation_config

    def model_evaluation(self, model_names, models):
        try:
            # On test set
            df = pd.read_csv(self.data_transformation_artifact.test_transform_file_path)
            X = df.drop(columns=["expenses", "Unnamed: 0"], axis=1)
            y = df[["expenses"]].squeeze()

            mae = {}
            mse = {}
            r2_score_values = {}
            for i in range(len(model_names)):
                model = models[i]
                mae[model_names[i]] = round(mean_absolute_error(y, model.predict(X)), 2)
                mse[model_names[i]] = round(mean_squared_error(y, model.predict(X)), 2)
                r2_score_values[model_names[i]] = round(
                    r2_score(y, model.predict(X)), 2
                )

            metrics = {"MAE": mae, "MSE": mse, "R2_SCORE": r2_score_values}

            # Best model onthe basis of R2 score on the test set
            best_model_key = max(r2_score_values, key=r2_score_values.get)
            best_model = models[model_names.index(best_model_key)]
            best_model_metrics = {
                "mae": mae[best_model_key],
                "mse": mse[best_model_key],
                "r2_score_value": r2_score_values[best_model_key],
            }

            return best_model, best_model_key, metrics, best_model_metrics, metrics

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self):
        try:
            df = pd.read_csv(
                self.data_transformation_artifact.train_transform_file_path
            )
            X = df.drop(columns=["expenses", "Unnamed: 0"], axis=1)
            y = df[["expenses"]].squeeze()

            with open(
                self.model_trainer_artifact.top_models_file_name_path, "r"
            ) as json_file:
                model_names = json.load(json_file)

            with open(self.model_trainer_artifact.model1_file_path, "rb") as f:
                model1 = pickle.load(f)

            with open(self.model_trainer_artifact.model2_file_path, "rb") as f:
                model2 = pickle.load(f)

            with open(self.model_trainer_artifact.model3_file_path, "rb") as f:
                model3 = pickle.load(f)

            with open(self.model_trainer_artifact.model4_file_path, "rb") as f:
                model4 = pickle.load(f)

            from sklearn.metrics import r2_score

            models = [model1, model2, model3, model4]
            mae = {}
            mse = {}
            r2_score_values = {}
            for i in range(len(model_names)):
                model = models[i]
                mae[model_names[i]] = round(mean_absolute_error(y, model.predict(X)), 2)
                mse[model_names[i]] = round(mean_squared_error(y, model.predict(X)), 2)
                r2_score_values[model_names[i]] = round(
                    r2_score(y, model.predict(X)), 2
                )

            metrics = {"MAE": mae, "MSE": mse, "R2_SCORE": r2_score_values}

            # Saving the train set metrics
            os.makedirs(
                self.model_evaluation_config.MODEL_EVALUATION_ARTIFACT_DIR,
                exist_ok=True,
            )
            with open(
                self.model_evaluation_config.TRAIN_METRICS_FILE_PATH, "w"
            ) as json_file:
                json.dump(metrics, json_file, indent=4)

            (
                best_model,
                best_model_name,
                test_metrics,
                best_model_metrics_test,
                metrics_test
            ) = self.model_evaluation(model_names, models)
            best_model_metrics_train = {
                "mae": mae[best_model_name],
                "mse": mse[best_model_name],
                "r2_score_value": r2_score_values[best_model_name],
            }

            best_model_metrics = {
                "Model": best_model_name,
                "Test Set": best_model_metrics_test,
                "Training Set": best_model_metrics_train,
            }
            
            # Track on the dagshub 
            # dagshub.init(repo_owner='ravikumar46931', repo_name='insurance-premium-MLOps', mlflow=True)

            mlflow.set_experiment("Model Evaluation")
            with mlflow.start_run(run_name='evaluate_best_model'):
                mlflow.log_param("Best Model", best_model_name)
                mlflow.log_metric("testset mae", best_model_metrics_test['mae'])
                mlflow.log_metric("testset mse", best_model_metrics_test['mse'])
                mlflow.log_metric("testset r2 score", best_model_metrics_test['r2_score_value'])
                mlflow.log_metric("trainset mae", best_model_metrics_train['mae'])
                mlflow.log_metric("trainset mse", best_model_metrics_train['mse'])
                mlflow.log_metric("trainset r2 score", best_model_metrics_train['r2_score_value'])
                mlflow.log_params(best_model.get_params())
                signature = infer_signature(X, best_model.predict(X))

                if best_model_name=="XGBoost":
                    mlflow.xgboost.log_model(
                        xgb_model=model,
                        artifact_path='best_ml_model',
                        signature=signature,
                        input_example=X.iloc[[0]],
                        registered_model_name="best_xgb_model"
                    )
                elif best_model_name=="LGB":
                    mlflow.lightgbm.log_model(
                        lgb_model=model,
                        artifact_path='best_ml_model',
                        signature=signature,
                        input_example=X.iloc[[0]],
                        registered_model_name="best_lgb_model",
                    )
                elif best_model_name=="CatBoost":
                    mlflow.catboost.log_model(
                        cb_model=model,
                        artifact_path='best_ml_model',
                        signature=signature,
                        input_example=X.iloc[[0]],
                        registered_model_name="best_catboost_model"
                    )

                else:
                    # Log the model
                    model_info = mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path=f"best_ml_model",
                        signature=signature,
                        input_example=X.iloc[[0]],
                        registered_model_name=f"best_{best_model_name}_model"
                    )




            # Saving the test set metrics
            with open(
                self.model_evaluation_config.TEST_METRICS_FILE_PATH, "w"
            ) as json_file:
                json.dump(test_metrics, json_file, indent=4)

            # Saving the best model metrics
            with open(
                self.model_evaluation_config.BEST_MODEL_METRICS_FILE_PATH, "w"
            ) as json_file:
                json.dump(best_model_metrics, json_file, indent=4)

            # Saving the best model
            with open(self.model_evaluation_config.BEST_MODEL_FILE_PATH, "wb") as file:
                pickle.dump(best_model, file)

            # Saving the best model to root
            os.makedirs("models", exist_ok=True)
            with open("models/best_model.pkl", "wb") as file:
                pickle.dump(best_model, file)

            # dagshub.init(repo_owner='ravikumar46931', repo_name='insurance-premium-MLOps', mlflow=True)


            # with mlflow.start_run():

            #     # Set a tag that we can use to remind ourselves what this run was for
            #     mlflow.set_tag("Evaluation Info", "This has model evaluation")


            model_evaluation_artifact = ModelEvaluationArtifact(
                self.model_evaluation_config.BEST_MODEL_FILE_PATH
            )

            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    df = pd.read_csv(r"artifact\Data_Transformation_Artifacts\transform.csv")
    X = df.drop(columns=["expenses", "Unnamed: 0"], axis=1)
    y = df[["expenses"]].squeeze()
    path = r"artifact\Model_Trainer_Artifacts\catboost_model.pkl"
    with open(path, "rb") as f:
        model4 = pickle.load(f)
    print(mean_absolute_error(y, model4.predict(X)))
    print(model4)
