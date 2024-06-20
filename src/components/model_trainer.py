import os
import sys
import json
import time
import warnings
import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import xgboost as xgb
import lightgbm as lgb
from src.logger import logging
from src.exception import CustomException
from src.constants import *
from src.constants.hyperparameter import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def fit_model_grid_search(self, model, cat_df, Y_df, model_name):
        try:
            warnings.filterwarnings("ignore")
            start_time = time.time()
            model.fit(cat_df, Y_df)
            best_params = model.best_params_
            best_model = model.best_estimator_
            print(f"Best hyperparameters for {model_name}:", best_params)
            print(f"Best model for {model_name}:{best_model}")
            end_time = time.time()
            print(f"Time taken for {model_name}: {(end_time-start_time):.2f} SECONDS\n")
            execution_time = round(end_time - start_time,2)
            return model, best_params, execution_time

        except Exception as e:
            raise CustomException(e, sys)

    def fit_model(self, model, cat_df, Y_df, model_name):
        try:
            warnings.filterwarnings("ignore")
            start_time = time.time()
            model.fit(cat_df, Y_df)
            end_time = time.time()
            print(f"Time taken for {model_name}: {(end_time-start_time):.2f} SECONDS\n")
            execution_time = round(end_time - start_time,2)
            return model, execution_time

        except Exception as e:
            raise CustomException(e, sys)

    def model_trainer(self, cat_df, Y_df):
        try:
            # Model initialization for Grid Search CV
            ridge = Ridge()
            lasso = Lasso()
            polynomial_pipe = make_pipeline(PolynomialFeatures(), LinearRegression())
            randomforest = RandomForestRegressor(random_state=0)
            gradientboost = GradientBoostingRegressor(random_state=0)

            # Models
            linreg = LinearRegression()
            xgbreg = xgb.XGBRegressor(**XGB_PARAMS)
            lgbreg = lgb.LGBMRegressor(**LGB_PARAMS)
            catboostreg = CatBoostRegressor(**CATBOOST_PARAMS)

            """Grid search cv"""
            ridge_grid_search = GridSearchCV(
                estimator=ridge, param_grid=RIDGE_PARAMS, cv=CV
            )
            lasso_grid_search = GridSearchCV(
                estimator=lasso, param_grid=LASSO_PARAMS, cv=CV
            )
            polynomial_grid_search = GridSearchCV(
                estimator=polynomial_pipe, param_grid=POLYNOMIAL_PARAMS, cv=CV
            )
            random_forest_grid_search = GridSearchCV(
                estimator=randomforest, param_grid=RANDOM_FOREST_PARAMS, cv=CV
            )
            gradient_boost_grid_search = GridSearchCV(
                estimator=gradientboost, param_grid=GRADIENTBOOST_PARAMS, cv=CV
            )

            tune_models_grid_search = {
                "Ridge": ridge_grid_search,
                "Lasso": lasso_grid_search,
                "Polynomial": polynomial_grid_search,
                "Random Forest": random_forest_grid_search,
                "Gradient Boost": gradient_boost_grid_search,
            }

            train_model_grid_search = {
                "Ridge": None,
                "Lasso": None,
                "Polynomial": None,
                "Random Forest": None,
                "Gradient Boost": None,
            }
            best_params_grid_search = {
                "Ridge": None,
                "Lasso": None,
                "Polynomial": None,
                "Random Forest": None,
                "Gradient Boost": None,
            }
            execution_time_grid_search = {
                "Ridge": None,
                "Lasso": None,
                "Polynomial": None,
                "Random Forest": None,
                "Gradient Boost": None,
            }

            for name, model in tune_models_grid_search.items():
                train_model_grid_search[name], best_params_grid_search[name], execution_time_grid_search[name] = self.fit_model_grid_search(
                    model, cat_df, Y_df, model_name=name
                )
               
            tune_model = {
                "Linear": linreg,
                "XGBoost": xgbreg,
                "LGB": lgbreg,
                "CatBoost": catboostreg,
            }

            train_model = {
                "Linear": None,
                "XGBoost": None,
                "LGB": None,
                "CatBoost": None,
            }
            train_model_execution_time = {
                "Linear": None,
                "XGBoost": None,
                "LGB": None,
                "CatBoost": None,
            }

            for name, model in tune_model.items():
                train_model[name], train_model_execution_time[name] = self.fit_model(model, cat_df, Y_df, model_name=name)
                
            return train_model_grid_search, train_model, best_params_grid_search, execution_time_grid_search, train_model_execution_time

        except Exception as e:
            raise CustomException(e, sys)

    def best_models(self, X, y, models):
        try:
            mae_val = {}
            for key, model in models.items():
                mae_val[key] = mean_absolute_error(y, model.predict(X))
                print(
                    f"{key} regression on training set MAE: {mean_absolute_error(y, model.predict(X)):.2f}"
                )

            mae_val = {
                k: v for k, v in sorted(mae_val.items(), key=lambda item: item[1])
            }
            top_models = dict(list(mae_val.items())[:NUMBER_OF_MODELS])

            print(f"The top {NUMBER_OF_MODELS} performing models are as follows: ")
            for name, mae_value in top_models.items():
                print(f"{name} regression on training set MAE: {mae_value}")

            return top_models, mae_val

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self):
        try:
            logging.info("Model traing started...")
            df = pd.read_csv(
                self.data_transformation_artifact.train_transform_file_path
            )
            X = df.drop(columns=["expenses", "Unnamed: 0"], axis=1)
            y = df[["expenses"]].squeeze()
            train_model_grid_search, train_model, best_params_grid_search, execution_time_grid_search, train_model_execution_time = self.model_trainer(X, y)
            models = {**train_model_grid_search, **train_model}
            top_models, mae_val = self.best_models(X, y, models)
            execution_time = {**execution_time_grid_search, **train_model_execution_time}
            
            # Model training tracking with mlflow

            dagshub.init(repo_owner='ravikumar46931', repo_name='insurance-premium-MLOps', mlflow=True)

            mlflow.set_experiment("Model Trainer")
            df_test = pd.read_csv(
                self.data_transformation_artifact.test_transform_file_path
            )
            X_test = df_test.drop(columns=["expenses", "Unnamed: 0"], axis=1)
            y_test = df_test[["expenses"]].squeeze()

            for name, model in models.items():
                with mlflow.start_run(run_name=name):
                    time = execution_time[name]
                    params = model.get_params()
                    mae = mae_val[name]
                    mlflow.log_metric("execution_time_sec", time)
                    mlflow.log_metric("mae", mae)
                    mlflow.log_params(params)

                    # Infer the model signature
                    signature = infer_signature(X_test, model.predict(X_test))

                    if name=="XGBoost":
                        mlflow.xgboost.log_model(
                            xgb_model=model,
                            artifact_path='xgb_ml_model',
                            signature=signature,
                            input_example=X_test.iloc[[0]],
                            registered_model_name="xgb_model"
                        )
                    elif name=="LGB":
                        mlflow.lightgbm.log_model(
                            lgb_model=model,
                            artifact_path='lgb_ml_model',
                            signature=signature,
                            input_example=X_test.iloc[[0]],
                            registered_model_name="lgb_model",
                        )
                    elif name=="CatBoost":
                        mlflow.catboost.log_model(
                            cb_model=model,
                            artifact_path='catboost_ml_model',
                            signature=signature,
                            input_example=X_test.iloc[[0]],
                            registered_model_name="catboost_model"
                        )

                    else:
                        # Log the model
                        model_info = mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path=f"{name}_ml_model",
                            signature=signature,
                            input_example=X_test.iloc[[0]],
                            registered_model_name=name
                        )

            # Save the model using pickle
            import pickle

            file_paths = []
            model_names = []
            os.makedirs(
                self.model_trainer_config.MODEL_TRAINER_ARTIFACT_DIR, exist_ok=True
            )
            os.makedirs(self.model_trainer_config.MODELS_DIR_PATH, exist_ok=True)
            for model_name, _ in top_models.items():
                name_model = "".join(model_name.split()).lower()
                file_name = name_model + "_model.pkl"
                file_path = os.path.join(
                    self.model_trainer_config.MODELS_DIR_PATH, file_name
                )
                with open(file_path, "wb") as f:
                    pickle.dump(models[model_name], f)
                file_paths.append(file_path)
                model_names.append(model_name)
            with open(
                self.model_trainer_config.TOP_MODELS_NAME_FILE_PATH, "w"
            ) as json_file:
                json.dump(model_names, json_file, indent=4)

            model_trainer_artifact = ModelTrainerArtifact(
                self.model_trainer_config.TOP_MODELS_NAME_FILE_PATH, *file_paths
            )

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
