# Model Card

## Model Details
- Created by Carlos López
- Date: April/26/2022
- Model: Support Vector Machine (as classifier)
- Trained using scikit-learn 

## Intended Use
The model can be used by persons that want to evaluate the earning of persons to offer a credit, just to mention a case of use as example. It was created to learn about the deployment process of a ML model.

## Training Data
The dataset used for this model is the Census Income Dataset available at https://archive.ics.uci.edu/ml/datasets/census+income
- Motivation: Know about the end-to-end process to deploy a ML model.
- Preprocessing: From the original dataset were removed the empty spaces and dataset was splitted randomly taking 80% as training set and 20% as test set. It was used a one hot encoder for input categorical variables and a label binarizer for the target variable (salary column).

## Evaluation Data
In the data there is some bias, such as sex and race bias. Fairness tests using Aequitas gave the following results:  
- Unsupervised Fairness: False.
- Supervised Fairness: False.
- Overall Fairness: False.

## Metrics
The metrics used to test the model were precision, recall and F1 score.
- Precision: 0.92
- Recall: 0.11
- F1 score: 0.19

## Ethical Considerations
This model was trained as example to know the process to deploy a ML model.
To know more about the dataset ethical aspects please go to it's web page.

## Caveats and Recommendations
In order for the experiment to be reproducible is needed to track the index of patterns tha are taken as test and training sets. Right now every time that the script runs, it splits the dataset.
