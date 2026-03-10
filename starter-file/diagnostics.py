
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('./starter-file/config.json','r') as f:
    config = json.load(f) 

##################Function to get model predictions
def model_predictions(dataset_df):
    #read the deployed model and a test dataset, calculate predictions

    prod_model = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl') 


    with open(prod_model, 'rb') as f:
        model = pickle.load(f)



    X = dataset_df.drop(['corporation', 'exited'], axis=1, errors='ignore')
    predictions = model.predict(X)

    return predictions.tolist() #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():

    dataset_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
            
    # Read the data
    df = pd.read_csv(dataset_path)
    numeric_columns = df.select_dtypes(include='number')

    #calculate summary statistics here

    means = numeric_columns.mean()

    medians = numeric_columns.median()
    standard_deviation = numeric_columns.std()


    # Convert each Series to a list
    mean_list = means.tolist()
    median_list = medians.tolist()
    std_list = standard_deviation.tolist()

    # Combine them into one single list
    summary_list = mean_list + median_list + std_list

    return summary_list #return value should be a list containing all summary statistics

##################Function count missing data
def missing_data():
    #  Load the data
    dataset_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
    df = pd.read_csv(dataset_path)
    
    # Calculate percentages of NA values per column
    # nas_count / total_rows * 100
    nas_percentages = (df.isna().sum() / len(df)) * 100
    
    # Return as a list
    return nas_percentages.tolist()


##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    #  Time the Ingestion script
    start_ingest = timeit.default_timer()
    os.system('python3 ./starter-file/ingestion.py')
    end_ingest = timeit.default_timer()
    ingestion_time = end_ingest - start_ingest

    # Time the Training script
    start_train = timeit.default_timer()
    os.system('python3 ./starter-file/training.py')
    end_train = timeit.default_timer()
    training_time = end_train - start_train

    return [ingestion_time, training_time]


##################Function to check dependencies
def outdated_packages_list():
    #get a list of outdated packages
    result = subprocess.run(['pip', 'list', '--outdated'], 
                            capture_output=True, 
                            text=True)
    print(result.stdout)
    return result.stdout


if __name__ == '__main__':
    dataset_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
    dataset_df = pd.read_csv(dataset_path)
    model_predictions(dataset_df)
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()





    
