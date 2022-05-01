import requests
import json

input_data = {
    "age": "31",
    "workclass": "Private",
    "fnlgt": "45781", 
    "education": "Masters",
    "education-num": "14",
    "marital-status":"Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital-gain": "14084",
    "capital-loss": "0",
    "hours-per-week": "50",
    "native-country": "United-States",
}

response = requests.post("http://127.0.0.1:8000/inference", 
                        data = json.dumps(input_data))

print(f'Status code: {response.status_code}')
print(f'Response: {response.json()}')