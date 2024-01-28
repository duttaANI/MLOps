"""MLOps Library"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import subprocess

# Specify the command to run the second script
command = ["python", "TITANIC.py"]

def logisticFunc(beta,X):
    #sigmoid
    return 1.0/(1 + np.exp(-np.dot(X,beta.T)))

def pred_value(beta, X):
    pred_prob = logisticFunc(beta,X)

    #print("pred_prob is:")
    #print(pred_prob)
    #print("shape of pred_prob:")
    #print(pred_prob.shape)
  
    return pred_prob


def load_beta():
    beta = np.loadtxt('beta.txt')
    return beta


def retrain():
    """Retrains the model
    """
    # Use subprocess to run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for the process to finish and capture the output
    stdout, stderr = process.communicate()


def format_input(x):
    """Takes int and converts to numpy array"""

    val = np.array(x)
    feature = val.reshape(-1, 1)
    return feature


def predict(Pclass,Fare,topic_id,Parch,SibSp,retrain):
    """Takes weight and predicts height"""
    
    
    #retrain()
    

    beta = load_beta()  # load_beta
    
    np_array_Pclass = format_input(Pclass)
    np_array_Fare = format_input(Fare)
    np_array_topic_id = format_input(topic_id)
    np_array_Parch = format_input(Parch)
    np_array_SibSp = format_input(SibSp)
    
    X_test = np.concatenate((np.array([[1]]),np_array_Pclass, np_array_Fare, np_array_topic_id, np_array_Parch, np_array_SibSp))
    X_test = X_test.astype(np.uint).ravel() # convert from 2d to 1d
    
    print("X_test.shape")
    print(X_test.shape)
    print(X_test.dtype)
   

    #X_test = np.hstack((np.matrix(np.ones(X_test.shape[0])).T, X_test))
    
    #print("X_test.shape")
    #print(X_test.shape)
    #print(X_test.head())
    print(X_test)
    
  
    
    y_final = pred_value(beta,X_test)
    y_prob = y_final*100
    
    print("y_prob")
    print(y_prob)
    predict_log_data = {
        "Pclass": np_array_Pclass,
        "Fare": np_array_Fare,
        "topic_id": np_array_topic_id,
        "Parch": np_array_Parch,
        "SibSp": np_array_SibSp,
        "survival probablity": y_prob,
    }
    logging.debug(f"Prediction: {predict_log_data}")
    payload = y_prob
    return payload
