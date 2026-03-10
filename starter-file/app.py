from flask import Flask, session, jsonify, request
import pandas as pd
from diagnostics import model_predictions, dataframe_summary, missing_data, execution_time, outdated_packages_list
from scoring import score_model
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('./starter-file/config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None

root_path = os.path.dirname(os.path.abspath(__file__))

#######################Prediction Endpoints
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    # Get data from body
    input_data = request.json
    file_path = input_data['filepath']
    dataset_df = pd.read_csv(os.path.join(root_path, file_path))
    prediction = model_predictions(dataset_df)
    return jsonify(prediction) #prediction_json #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():        
    #check the score of the deployed model
    stats = score_model()
    return jsonify(stats) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary_stats():
    stats = dataframe_summary()       
    #check means, medians, and modes for each column
    return jsonify(stats) #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostic_stats():        
    #check timing and percent NA values
    time =execution_time()
    missing = missing_data()       
    outdated = outdated_packages_list()

    diagnostics_dict = {
        "execution_time": time,
        "missing_data": missing,
        "outdated_packages": outdated
    }

    return jsonify(diagnostics_dict) #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
