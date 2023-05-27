import logging
import warnings

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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



    # Charger le data
    df = pd.read_csv('pointure.data')
    # df

    # df.describe()

    # pretraitement des donnees


    label_encoder = preprocessing.LabelEncoder()
    input_classes = ['masculin', 'féminin']
    label_encoder.fit(input_classes)

    # transformer un ensemble de classes
    encoded_labels = label_encoder.transform(df['Genre'])
    print(encoded_labels)
    df['Genre'] = encoded_labels

    # df

    # Séparation des features et de la variable cible
    X = df.iloc[:, lambda df: [1, 2, 3]]
    y = df.iloc[:, 0]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

    # print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Définir le modèle
    gnb = GaussianNB()

    # Entraînement du modèle

    gnb.fit(X_train, y_train)

    # EVALUATION SUR LE TRAIN
    y_naive_bayes1 = gnb.predict(X_train)
    print("Number of mislabeled points out of a total 0%d points : 0%d" % (
    X_train.shape[0], (y_train != y_naive_bayes1).sum()))

    accuracy = metrics.accuracy_score(y_train, y_naive_bayes1)
    print("Accuracy du modele Naive Bayes predit: " + str(accuracy))

    recall_score = metrics.recall_score(y_train, y_naive_bayes1)
    print("recall score du modele Naive Bayes predit: " + str(recall_score))

    f1_score = metrics.f1_score(y_train, y_naive_bayes1)
    print("F1 score du modele Naive Bayes predit: " + str(f1_score))

    d = {'Taille(cm)': [183], 'Poids(kg)': [59], 'Pointure(cm)': [20]}
    dfToPredict = pd.DataFrame(data=d)
    # dfToPredict

    yPredict = gnb.predict(dfToPredict)
    print('La classe predite est : ', yPredict)

    y_pred = gnb.predict(X_test)

    # Évaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print("Accuracy: {}".format(accuracy))
    print("recall score: {}".format(recall_score))
    print("F1 score: {}".format(f1_score))

    # Écrire les scores dans un fichier
    with open("metrics.txt", 'w') as outfile:
        outfile.write("Accuracy: {}".format(accuracy))
        outfile.write("recall score: {}\n".format(recall_score))
        outfile.write("F1 score: {}\n".format(f1_score))





