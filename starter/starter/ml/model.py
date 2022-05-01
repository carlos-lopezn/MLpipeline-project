"""
Module with functions to perform the training, inference and computation of
metricsa ML model
"""
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.svm import LinearSVC

# Optional: implement hyperparameter tuning.
def train_model(x_train, y_train):
    """
    Trains a Support Vector Machine (SVM) model and returns it.

    Inputs
    ------  
    x_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained SVM model.
    """
    svc_model = LinearSVC(max_iter=1000)
    svc_model.fit(x_train, y_train)

    return svc_model

def compute_model_metrics(targets, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    targets : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(targets, preds, beta=1, zero_division=1)
    precision = precision_score(targets, preds, zero_division=1)
    recall = recall_score(targets, preds, zero_division=1)

    return precision, recall, fbeta


def inference(model, inputs):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Support Vector Machine as regressor
        Trained machine learning model.
    inputs : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(inputs)
