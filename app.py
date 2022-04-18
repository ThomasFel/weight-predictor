from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def weight_prediction():
    if request.method == 'GET':
        return render_template("index.html")
    
    elif request.method == 'POST':
        clf = joblib.load("weight-predictor-using-linear-regression.pkl")

        height = float(request.form.get("height"))
        gender = int(request.form["options"])

        result = clf.predict([[gender, height]])
        result2 = np.array_str(result, suppress_small=True)
        result3 = float(result2.translate({ord(i): None for i in '[]'}))

        return render_template("index.html", result = float("{:.2f}".format(result3)))

    else:
        return "Unsupported Request Method."

if __name__ == '__main__':
    app.run(port=5000, debug=True)