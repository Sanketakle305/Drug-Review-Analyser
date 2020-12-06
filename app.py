# importing library
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Loading models
Linear_svc = joblib.load('models/linear_svc.pkl')
tfidf_5000 = joblib.load('models/Tfidf_vector.pkl')


app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/home", methods=['POST', 'GET'])
def back_home():
    return render_template('index.html')


@app.route("/results", methods=['POST', 'GET'])
def resuts():
    if request.method == 'POST':
        message = request.form['textbox']
        X_val = tfidf_5000.transform([message]).toarray()
        val_pred = Linear_svc.predict(X_val)

        if int(val_pred) == 1:
            return render_template('Positive.html')
        else:
            return render_template('Negative.html')


if __name__ == "__main__":
    app.run()
