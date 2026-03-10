import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions



###############Load config.json and get path variables
with open('./starter-file/config.json','r') as f:
    config = json.load(f) 


##############Function for reporting
def score_model(dataset_df):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    # get predicted values
    predicted_values = model_predictions(dataset_df)

    # actual values
    actual_values = dataset_df['exited']

    # confusion matrix
    confusion_matrix = metrics.confusion_matrix(actual_values, predicted_values)

    # plot matrix
    display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
    display.plot()

    # saving plot to workspace
    output_path = os.path.join(config['output_model_path'], 'confusionmatrix.png')
    plt.savefig(output_path)


if __name__ == '__main__':
    # load data
    test_dataset_csv_path = os.path.join(config['test_data_path'], 'testdata.csv')
    test_data_df = pd.read_csv(test_dataset_csv_path)
    score_model(test_data_df)
