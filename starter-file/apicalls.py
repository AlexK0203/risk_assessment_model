import requests

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"



#Call each API endpoint and store the responses
# prediction endpoint
response1 = requests.post(URL + "prediction", json={"filepath":"testdata/testdata.csv"}).text
# scoring endpoint
response2 = requests.get(URL + "scoring").text
# summary endpoint
response3 = requests.get(URL + "summarystats").text
# diagnostics endpoint
response4 = requests.get(URL + "diagnostics").text

#combine all API responses
with open("apireturns.txt", "w") as f:
    f.write(f"prediction: {response1}\n")
    f.write(f"scoring: {response2}\n")
    f.write(f"summarystats: {response3}\n")
    f.write(f"diagnostics: {response4}\n")

#write the responses to your workspace



