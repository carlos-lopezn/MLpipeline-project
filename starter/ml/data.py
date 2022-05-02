"""
Module to process data that feeds a ML model
"""
import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def load_transform(transform_path):
    """
    load a scikit transformation as .sav file, it can be a model or a encoder

    input:
        transform_path: (string) path to the transformation with .sav extension
    returns:
        transform: sckit model or encoder
    """
    with open(transform_path, 'rb') as transform_file:
        transform = pickle.load(transform_file)

    return transform


def save_transform(transform_path, transform_object):
    """
    save a scikit transformation as .sav file, it can be a model or a encoder

    input:
        transform_path: (string) path to store the model as .sav file
        transform_object: (sckit model) the scikit model thta we want to store
    returns:
        None
    """
    with open(transform_path, 'wb') as transform_file:
        pickle.dump(transform_object, transform_file)


def process_data(
        dataset_df,
        categorical_features=[],
        label=None,
        training=True,
        encoder=None,
        lb=None):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    dataset_df : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        dataset_df,
        Name of the label column in `dataset_df`. If None, then an empty array will be returned
        for targets (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    inputs : np.array
        Processed data.
    targets : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        targets = dataset_df[label]
        inputs = dataset_df.drop([label], axis=1)
    else:
        targets = np.array([])

    x_categorical = inputs[categorical_features].values
    x_continuous = inputs.drop(*[categorical_features], axis=1)

    if training is True:
        encoder_path = os.path.join(os.getcwd(), '..', 'model', 'encoder.sav')
        binarizer_path = os.path.join(
            os.getcwd(), '..', 'model', 'label_binarizer.sav')
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        x_categorical = encoder.fit_transform(x_categorical)
        targets = lb.fit_transform(targets.values).ravel()
        save_transform(encoder_path, encoder)
        save_transform(binarizer_path, lb)

    else:
        if encoder is not None:
            encoder = load_transform(encoder)
        else:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        if lb is not None:
            lb = load_transform(lb)
        else:
            lb = LabelBinarizer()

        x_categorical = encoder.transform(x_categorical)
        try:
            targets = lb.transform(targets.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    inputs = np.concatenate([x_continuous, x_categorical], axis=1)

    return inputs, targets, encoder, lb


def split_slices(test_df, slice_column):
    """ Split the test dataset in slices.

    Inputs
    ------
    test_df : (pandas dataframe) test dataset
    slice_colum : (string) the name of the column to slice
    Returns
    -------
    slices : (dictionary) dictionary with a pandas dataframe for each category
                in sliced column
    """
    slices = {}
    for category in test_df[slice_column].unique():
        slice_test = test_df[test_df[slice_column] == category].copy()
        slices[category] = slice_test

    return slices
