import training
import scoring
import deployment
import diagnostics
import reporting
import json
import os
import ingestion
import pandas as pd

# Load configurations
with open('./starter-file/config.json','r') as f:
    config = json.load(f)

source_data_path = config['input_folder_path']
ingested_log_path = os.path.join(config['prod_deployment_path'], 'ingestedfiles.txt')

all_files = os.listdir(source_data_path)

##################Check and read new data
#first, read ingestedfiles.txt

with open(ingested_log_path, 'r') as f:
    ingested_log = f.read().splitlines()

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt

new_files = [file for file in all_files if file not in ingested_log and file.endswith('.csv')]

# Check if we found anything new
if len(new_files) > 0:
    print(f"Found {len(new_files)} new files: {new_files}")
    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here  
    # Trigger the ingestion script
    ingestion.merge_multiple_dataframe()
else:
    print("No new data found. Ending process.")
    exit()


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data

latest_score_file = os.path.join(config['prod_deployment_path'], 'latestscore.txt')
with open(latest_score_file, 'r') as f:
    old_score = float(f.read())



##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here

new_data_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
new_score = scoring.score_model(path=new_data_path)

print(f"Old Score: {old_score}")
print(f"New Score: {new_score}")

if new_score < old_score:
    print(f"Model drift detected! (Old: {old_score}, New: {new_score})")
    
    # 1. Re-training
    print("Re-training model...")
    training.train_model()
    
    # 2. Re-deployment
    print("Re-deploying model...")
    deployment.store_model_into_pickle()
    
    # 3. Diagnostics and Reporting
    print("Running final diagnostics and reporting...")
    # Load the new data for the diagnostics functions

    current_data = pd.read_csv(new_data_path)
    
    # Pass the dataframe to the prediction function
    predictions = diagnostics.model_predictions(current_data) 
    
    # Run the rest
    diagnostics.dataframe_summary()
    diagnostics.missing_data()

    times = diagnostics.execution_time()
    print(f"Execution times: Ingestion={times[0]:.2f}s, Training={times[1]:.2f}s")
    
    reporting.score_model(current_data)  
    print("Process complete: Model updated.")


    os.system('python ./starter-file/apicalls.py') #run apicalls.call_api_endpoints()
    print("Process complete: Model updated and API results recorded.")

else:
    print(f"No model drift detected (New score {new_score} >= Old score {old_score}).")