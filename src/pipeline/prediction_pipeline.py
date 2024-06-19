import sys
import joblib
import pickle
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging


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
