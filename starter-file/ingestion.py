import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    content = os.listdir(input_folder_path)
    df_list = []
    ingested_filenames = []
    for file in content:
        if file.endswith('.csv'):
            # Path to the specific file
            path = os.path.join(input_folder_path, file)
            
            # Read the data
            df = pd.read_csv(path)
            
            # 3. Save the data AND the filename separately
            df_list.append(df)
            ingested_filenames.append(file)

    # Combine and remove Duplicates
    if len(df_list) > 0:
            # 1. Combine and remove Duplicates
            final_df = pd.concat(df_list).drop_duplicates()

            # 2. Save CSV
            csv_path = os.path.join(output_folder_path, 'finaldata.csv')
            final_df.to_csv(csv_path, index=False)
            
            # 3. Save Record
            txt_path = os.path.join(output_folder_path, 'ingestedfiles.txt')
            with open(txt_path, 'w') as f:
                f.write('\n'.join(ingested_filenames))
                
            return final_df
    
    else:
        print("No new CSV files found. Skipping save.")
        return None # Or an empty DataFrame, depending on your preference


if __name__ == '__main__':
    merge_multiple_dataframe()
