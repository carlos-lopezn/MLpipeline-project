"""
Module with the API to perform the predictions of a ML model
"""
# Put the code for your API here.
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import starter
from starter.ml.data import process_data, load_transform
from starter.ml.model import inference

encoder_path = os.path.join(
                os.path.dirname(os.path.abspath((starter.__file__))),
                '..',
                'model',
                'encoder.sav')
binarizer_path = os.path.join(
                os.path.dirname(os.path.abspath((starter.__file__))),
                '..',
                'model',
                'label_binarizer.sav')
model_path = os.path.join(
                os.path.dirname(os.path.abspath((starter.__file__))),
                '..',
                'model',
                'svc_model.sav')
                
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

app = FastAPI()


class InputData(BaseModel):
    """
    Class tha inherits from BaseModel to specify the data schema
    """
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        """
        Class to configure the example that is shown in the docs
        """
        schema_extra = {
            "example": {
                "age": 20,
                "workclass": "Private",
                "fnlgt": 266015,
                "education": "Some-college",
                "education-num": 10,
                "marital-status": "Never-married",
                "occupation": "Sales",
                "relationship": "Own-child",
                "race": "Black",
                        "sex": "Male",
                        "capital-gain": 0,
                        "capital-loss": 0,
                        "hours-per-week": 44,
                        "native-country": "United-States",
            }
        }


@app.get("/")
async def greetings():
    """
    Get method to return a message
    """
    return {"message": "Greetings human!"}


@app.post("/inference")
async def perform_inference(input_data: InputData):
    """
    Post method to perform an inference over the data in the request body
    """
    input_data_dict = input_data.dict()
    # We add the column salary in order for dataframe to have the same
    # shape expected by process_data function
    input_data_dict['salary'] = '>50K'

    for key, value in input_data_dict.items():
        input_data_dict[key] = [value]

    pattern_df = pd.DataFrame.from_dict(input_data_dict)

    columns = [key.replace('_', '-') for key, _ in input_data_dict.items()]
    pattern_df.columns = columns
    x_test, _, _, _ = process_data(
        pattern_df,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder_path,
        lb=binarizer_path)
    model = load_transform(model_path)
    prediction = inference(model, x_test)

    return {'prediction': int(prediction)}
