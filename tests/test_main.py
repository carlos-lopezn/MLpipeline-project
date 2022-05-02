"""
Module to test the main app script
"""
import requests
import json

def test_inference_post_1():
    """
    Testing post at '/inference' when the input data throws 1 as prediction
    (Persons who's salary is >50K)
    """
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
    assert response.status_code == 200
    assert response.json()['prediction'] == 1

def test_inference_post_0():
    """
    Testing post at '/inference' when the input data throws 0 as prediction
    (Persons who's salary is <=50K)
    """
    input_data = {
        "age": "20",
        "workclass": "Private",
        "fnlgt": "266015", 
        "education": "Some-college",
        "education-num": "10",
        "marital-status":"Never-married",
        "occupation": "Sales",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital-gain": "0",
        "capital-loss": "0",
        "hours-per-week": "44",
        "native-country": "United-States",
    }
    response = requests.post("http://127.0.0.1:8000/inference", 
                            data = json.dumps(input_data))
    assert response.status_code == 200
    assert response.json()['prediction'] == 0 

def test_get():
    """
    Testing get at '/' to obtain a message response
    """
    response = requests.get("http://127.0.0.1:8000/")
    assert response.status_code == 200
    assert response.json()['message'] == 'Greetings human!'