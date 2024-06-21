from sklearn.model_selection import train_test_split
from MessageToVector import MessageToVector
from LogisticRegression import LogisticRegression
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
from OneVsRestLogisticRegression import OneVsRestLogisticRegression
from sklearn import datasets


def preparing_data_and_model():
    df = pd.read_csv("spam_ham_dataset.csv")
    X = df.iloc[:, 2].values
    y = df.iloc[:, -1].values
    message_to_vector = MessageToVector()
    vector_X = message_to_vector.fit_transform_to_reduced(X)
    vector_X = np.c_[vector_X, np.ones(vector_X.shape[0])] # add constant
    X_train, X_test, y_train, y_test = train_test_split(vector_X, y, test_size=0.2, random_state=42)

    lg = LogisticRegression()
    lg.fit(X_train, y_train)
    return lg, X_train, X_test, y_train, y_test

def first_and_second_question():
    print("-------First and Second questions-------")
    lg, X_train, X_test, y_train, y_test = preparing_data_and_model()
    lg.fit(X_train, y_train)
    print("The Weights: ", lg.get_weights())
    print("The Score: ", lg.score(X_test, y_test))

def third_question():
    print("-------Third question-------")
    lg, X_train, X_test, y_train, y_test = preparing_data_and_model()
    lr_probs = lg.predict_proba(X_test)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    plt.plot(lr_fpr, lr_tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("The ROC curve for the model")
    # show the plot
    plt.show()

def fourth_question():
    print("-------fourth question-------")
    iris = datasets.load_iris()
    X = iris.data
    X = np.c_[X, np.ones(X.shape[0])] # add constant
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
    multiclass_classification = OneVsRestLogisticRegression()
    multiclass_classification.fit(X_train, y_train)
    test_predictions = multiclass_classification.predict(X_test)
    print("The real values:     ", y_test)
    print("The predicted values:", test_predictions)
    print("The Score: ", multiclass_classification.score(X_test, y_test))



if __name__ == "__main__":
    first_and_second_question()
    third_question()
    fourth_question()



