# ML pipeline project

The goal of the project is to learn how to implement an end-to-end ML pipeline solution. The dataset used to train a Support Vector Machine as classifier is the Census Income Data Set.

## Dependencies
> - numpy
> - pandas
> - scikit-learn
> - pytest
> - requests
> - fastapi
> - uvicorn
> - gunicorn
> - aequitas

## How to use it
In order to perform the predictions, it is required to run first the web server using the command
> uvicorn main:app --reload  

Then you can perform post request using the API documentation located at http://localhost:8000/docs