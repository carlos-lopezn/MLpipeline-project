"""
Module to test the functions in the model module
"""
import os
import pytest
import pandas as pd
from sklearn.svm import LinearSVC
from ..model import train_model, inference, compute_model_metrics
from ..data import load_transform, process_data


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


@pytest.fixture(scope='module')
def inputs_np():
    """
    fixture to generate the numpy array with the inputs to feed the model

    inputs:
        None
    returns:
        x_dataset: (numpy) the input patterns for the model
    """
    dataset_df = pd.read_csv(os.path.join(
        os.getcwd(),
        'starter',
        'data',
        'cleaned_census.csv'
        )
    )

    encoder_path = os.path.join(
        os.getcwd(),
        'starter',
        'model',
        'encoder.sav'
    )

    x_dataset, _, _, _ = process_data(dataset_df,
                                      categorical_features=cat_features,
                                      label="salary",
                                      training=False,
                                      encoder=encoder_path)

    return x_dataset


@pytest.fixture(scope='module')
def targets_np():
    """
    fixture to generate the targets or labels for the input patterns

    inputs:
        None
    returns:
        y_dataset: (numpy) targets or labels for the input patterns
    """
    dataset_df = pd.read_csv(os.path.join(
        os.getcwd(),
        'starter',
        'data',
        'cleaned_census.csv'
    )
    )

    encoder_path = os.path.join(
        os.getcwd(),
        'starter',
        'model',
        'encoder.sav'
    )

    binarizer_path = os.path.join(
        os.getcwd(),
        'starter',
        'model',
        'label_binarizer.sav'
    )

    _, y_dataset, _, _ = process_data(dataset_df,
                                      categorical_features=cat_features,
                                      label="salary",
                                      training=False,
                                      encoder=encoder_path,
                                      lb=binarizer_path)

    return y_dataset


@pytest.fixture(scope='module')
def model():
    """
    fixture to load the model

    inputs:
        None
    returns
        model: (Scikit class) the stored model previously trained
    """
    model_path = os.path.join(
                    os.getcwd(),
                    'starter',
                    'model',
                    'svc_model.sav'
                 )

    return load_transform(model_path)


def test_train_model(inputs_np, targets_np):
    """
    testing on train_model function
    """
    assert isinstance(train_model(inputs_np, targets_np), LinearSVC)


def test_inference(targets_np, model, inputs_np):
    """
    testing on inference function
    """
    preds = inference(model, inputs_np)

    assert targets_np.shape[0] == preds.shape[0]
    assert preds.all() is not None


def test_compute_model_metrics(targets_np, model, inputs_np):
    """
    testing on compute_model_metrics function
    """
    preds = inference(model, inputs_np)
    precision, recall, f1score = compute_model_metrics(targets_np, preds)

    assert precision is not None
    assert recall is not None
    assert f1score is not None
