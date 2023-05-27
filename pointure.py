# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
# ! pip install --user mlflow

import logging
import warnings

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL

    try:
        df = pd.read_csv('pointure.data')
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e)

    label_encoder = preprocessing.LabelEncoder()
    input_classes = ['masculin', 'féminin']
    label_encoder.fit(input_classes)

    # transformer un ensemble de classes
    encoded_labels = label_encoder.transform(df['Genre'])
    # print(encoded_labels)
    df['Genre'] = encoded_labels

    # Split the data into training and test sets. (0.75, 0.25) split.
    X = df.iloc[:, lambda df: [1, 2, 3]]
    y = df.iloc[:, 0]

    # The predicted column is "quality" which is a scalar from [3, 9]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

    # Note de

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name='experiment2')

    with mlflow.start_run():
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_naive_bayes1 = gnb.predict(X_train)

        accuracy = metrics.accuracy_score(y_train, y_naive_bayes1)
        print("Accuracy du modele Naive Bayes predit: " + str(accuracy))

        recall_score = metrics.recall_score(y_train, y_naive_bayes1)
        print("recall score du modele Naive Bayes predit: " + str(recall_score))

        f1_score = metrics.f1_score(y_train, y_naive_bayes1)
        print("F1 score du modele Naive Bayes predit: " + str(f1_score))

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall_score", recall_score)
        mlflow.log_metric("f1_score", f1_score)

        mlflow.sklearn.log_model(gnb, "model")