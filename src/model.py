"""Created by Vishak

Datetime: 17-05-2023

Description: Helper file to load trained model and perform predictions 
"""

import pandas as pd 
import pickle
from sklearn.preprocessing import MinMaxScaler

SCALER_SAVE_PATH = "../models/saved/scaler_model.sav"
SVC_MODEL_PATH = "../models/saved/svc_model.pkl"

def scale_data(test_data):
    """Scale data using Sklearn MinMaxScaler

    Args:
        test_data (array/list): List of input values for test

    Returns:
        array/list: Returns test data after performing MinMaxScaling operation
    """

    with open(SCALER_SAVE_PATH, 'rb') as scaler_model:
        scaler = pickle.load(scaler_model)

    scaled_test_data = scaler.transform(test_data)

    return scaled_test_data


def load_model():
    """Load serialized ML model (.pkl)
    """

    with open(SVC_MODEL_PATH, 'rb') as svc_model_file:
        svc_model = pickle.load(svc_model_file)

    
    return svc_model

def predict(test_data):
    """Predict on test data - based on saved model

    Args:
        test_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    labels = {0:'No Disease', 1: 'Disease'}

    scaled_data = scale_data(test_data)
    
    svc_model = load_model()
        
    oos_predict = svc_model.predict(scaled_data)
    print("predict", oos_predict)

    # print("Feature names",svc_model.feature_names)
    return labels[oos_predict[0]]