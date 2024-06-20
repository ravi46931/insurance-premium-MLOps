import sys
import os
import joblib
import pickle
import json
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.constants import ARTIFACT_DIR, PREDICTION_PATH, PREDICTION_FILE
import warnings


class PredictionPipeline:
    def __init__(self):
        pass

    def prediction(self, data_point):
        try:
            preprocessor = joblib.load("models/preprocessor.joblib")
            # model = pickle.load('model/best_model.pkl')
            with open("models/best_model.pkl", "rb") as file:
                model = pickle.load(file)

            # single_df = pd.DataFrame(single_data_point)
            single_df = pd.DataFrame(data_point[0])
            tr_data = preprocessor.transform(single_df)

            data_val = np.append(tr_data, data_point[1])
            data_val = data_val.reshape(1, -1)

            predicted_value = model.predict(data_val)

            return predicted_value[0]

        except Exception as e:
            raise CustomException(e, sys)


def main():
    warnings.filterwarnings('ignore')
    REGION = "southwest"
    SEX = "male"
    SMOKER = "no"
    AGE = 15
    BMI = 19.7
    CHILDREN = 3
    pred_pipeline = PredictionPipeline()
    single_data_point = {
            "region": [REGION],
            "sex": [SEX],
            "smoker": [SMOKER],
            "age": [AGE],
            "bmi": [BMI]
        }
    data = [single_data_point, CHILDREN]
    predicted_val = pred_pipeline.prediction(data)
    pred_path = os.path.join(ARTIFACT_DIR, PREDICTION_PATH)
    os.makedirs(pred_path, exist_ok=True)
    dict_data = {
        'INPUT VALUE': {
            "region": REGION,
            "sex": SEX,
            "smoker": SMOKER,
            "age": AGE,
            "bmi": BMI,
            "children": CHILDREN
        },
        "PREDICTED VALUE": predicted_val
    }

    with open(os.path.join(pred_path, PREDICTION_FILE), 'w') as f:
        json.dump(dict_data, f, indent=4)

    print(predicted_val)

if __name__=="__main__":
    main()