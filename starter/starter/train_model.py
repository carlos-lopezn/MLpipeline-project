"""
Module to train a support vector classifier
"""
# Script to train machine learning model.
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from aequitas.plotting import Plot
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from ml.data import process_data, load_transform, save_transform, split_slices
from ml.model import train_model, compute_model_metrics, inference

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

ap = Plot()

data = pd.read_csv(
    os.path.join(
        os.getcwd(),
        '..',
        'data',
        'cleaned_census.csv'))


def bias_fairness_report(test_df, y_test, preds):
    """
    Using Aequitas to obtain fairness information
    inputs:
        test_df: (dataframe) dataframe with the test set
        y_test: (numpy) targets or labels from the test set
        preds: (numpy) predictions made by the model
    returns:
        None
    """
    df_aq = test_df.copy()[cat_features]
    df_aq['label_value'] = y_test
    df_aq['score'] = preds

    group = Group()
    xtab, _ = group.get_crosstabs(df_aq)

    bias = Bias()
    bias_df = bias.get_disparity_major_group(
        xtab, original_df=df_aq, alpha=0.05, mask_significance=True)

    fairness = Fairness()
    fairness_df = fairness.get_group_value_fairness(bias_df)
    overall_fairness = fairness.get_overall_fairness(fairness_df)

    for key, value in overall_fairness.items():
        print(f"{key}: {value}")
    print("----------------------------------------------")


def prediction_on_slice(slice_column, test_df, model, encoder, binarizer):
    """
    funtion to obtain the model performance over a slice of data

    inputs:
        slice_column: (string) name of the column to slice
        test_df: (dataframe) test set
        model: (scikit class) trained ML model
        encoder: (scikit class) encoder used process inputs
        binarizer: (scikit class) label binarizer for the targets
    returns:
        None

    """
    # Slice test set dependending on name of the column and get performance
    # over each slice
    if slice_column is not None and slice_column in cat_features:
        print("----------------------------------------------")
        print(f"Performance over {slice_column} slice")
        print("----------------------------------------------")
        slices = split_slices(test_df, slice_column)
        for key, _ in slices.items():
            x_test_slice, y_test_slice, _, _ = process_data(
                slices[key],
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=binarizer
            )
            preds_slice = inference(model, x_test_slice)

            precision_slice, recall_slice, fbeta_slice = compute_model_metrics(
                y_test_slice, preds_slice)

            print(f"Number of patterns grouped by {key}: {len(x_test_slice)}")
            print(f"Precision grouped by {key}: {precision_slice}")
            print(f"Recall grouped by {key}: {recall_slice}")
            print(f"F1 score grouped by {key}: {fbeta_slice}")
            print("----------------------------------------------")

# Add the necessary imports for the starter code.


def main(training=True, slice_column=None):
    """
    main function of the scipt that trains a ML model to classify
    data depending on people's salary. Produces a summary of the performance overall
    or over slices of data.

    Input:
        training: (bool) If true the ML is trained, otherwise the model is
                         loaded from the model folder
        slice_column: (string) Name of the column in the dataset to slice

    Returns:
        None
    """
    # Add code to load in the data.

    print("Splitting dataset.")
    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.
    train, test = train_test_split(data, test_size=0.20)

    if training:
        print("Preprocessing dataset and training model.")
        x_train, y_train, encoder, lb = process_data(
            train, categorical_features=cat_features, label="salary"
        )

        x_test, y_test, _, _ = process_data(
            test,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb
        )

        model = train_model(x_train, y_train)

        model_path = os.path.join(os.getcwd(), '..', 'model', 'svc_model.sav')
        save_transform(model_path, model)

    else:
        print("Preprocessing dataset and loading model.")
        _, _, encoder, lb = process_data(
            train, categorical_features=cat_features, label="salary", training=False)

        x_test, y_test, _, _ = process_data(
            test,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb
        )

        model_path = os.path.join(os.getcwd(), '..', 'model', 'svc_model.sav')
        model = load_transform(model_path)

    preds = inference(model, x_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print("----------------------------------------------")
    print(f"Overall performance ({len(x_test)} test patterns)")
    print("----------------------------------------------")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {fbeta}\n")

    bias_fairness_report(test, y_test, preds)

    prediction_on_slice(slice_column, test, model, encoder, lb)


if __name__ == '__main__':
    main()
