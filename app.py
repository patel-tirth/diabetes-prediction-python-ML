import flask
from flask import Flask, render_template, flash
from flask import request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = "something secret"


@app.route("/", methods=['POST', 'GET'])
def show_form():
    if request.method == 'POST':
        predictionScript(request.form['pregnancies'], request.form['glucose'], request.form['bloodpressure'],
                         request.form['skinthickness'],
                         request.form['insulin'], request.form['bmi'], request.form['dpf'], request.form['age'])
    return render_template("index.html")


def predictionScript(pg, glucose, bp, st, insulin, bmi, dpf, age):
    # data collection and analysis
    # pima diabetes dataset

    diabetes_dataset = pd.read_csv('./diabetes.csv')

    # print(diabetes_dataset.head())

    # getting statistical measures
    # print(diabetes_dataset.describe())

    # print(diabetes_dataset['Outcome'].value_counts())

    # print(diabetes_dataset.groupby('Outcome').mean())

    # separate outcome column from other columns

    X = diabetes_dataset.drop(columns='Outcome', axis=1)  # drop column
    Y = diabetes_dataset['Outcome']

    # data standardization

    scaler = StandardScaler()

    scaler.fit(X)

    standardized_data = scaler.transform(X)

    # print(standardized_data)
    X = standardized_data

    # train test split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # print(X.shape, X_train.shape, X_test.shape)

    # training the model

    classifier = svm.SVC(kernel='linear')

    # training support vector machine classifier
    classifier.fit(X_train, Y_train)

    # model evaluation

    # Accuracy score on training data

    X_train_prediction = classifier.predict(X_train)

    # compare x train with original y train table
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    print('Accuracy score of training data: ', training_data_accuracy)

    # Accuracy score on test data
    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    print('Accuracy score of test data: ', test_data_accuracy)

    # making a predictive system
    input_data = (pg, glucose, bp, st, insulin, bmi, dpf, age)

    input_data_as_numpy_array = np.asarray(input_data)

    # reshaping the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    std_data = scaler.transform(input_data_reshaped)

    print(std_data)

    prediction = classifier.predict(std_data)

    if prediction[0] == 1:
        flash('You may  be diabetic', category="flashMsgDiabetic")
        print("Patient is diabetic")
    else:
        flash('You may not be diabetic', category="flashMsgNotDiabetic")
        print("Patient is not diabetic")


# @app.route('/result', methods=['POST'])
# def result():

if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=3000)
