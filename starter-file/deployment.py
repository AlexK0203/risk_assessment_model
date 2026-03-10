from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil


##################Load config.json and correct path variable
with open('./starter-file/config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
score_path = os.path.join(config['output_model_path'], 'latestscore.txt')
ingest_path = os.path.join(config['output_folder_path'], 'ingestedfiles.txt')
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    
    if not os.path.exists(prod_deployment_path):
        os.makedirs(prod_deployment_path)

    # copy model
    shutil.copy(model_path, prod_deployment_path)

    # copy score
    shutil.copy(score_path, prod_deployment_path)

    # copy ingest files
    shutil.copy(ingest_path, prod_deployment_path)

if __name__ == '__main__':
    store_model_into_pickle()
