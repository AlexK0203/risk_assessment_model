import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

# Load config
with open('./starter-file/config.json','r') as f:
    config = json.load(f)

#Call each API endpoint and store the responses
# prediction endpoint
response1 = requests.post(URL + "prediction", json={"filepath":"testdata/testdata.csv"}).text
# scoring endpoint
response2 = requests.get(URL + "scoring").text
# summary endpoint
response3 = requests.get(URL + "summarystats").text
# diagnostics endpoint
response4 = requests.get(URL + "diagnostics").text

# Get the models folder path
output_path = os.path.join(config['output_model_path'], 'apireturns.txt')

#combine all API responses
with open(output_path, "w") as f:
    f.write(f"prediction: {response1}\n")
    f.write(f"scoring: {response2}\n")
    f.write(f"summarystats: {response3}\n")
    f.write(f"diagnostics: {response4}\n")

#write the responses to your workspace



