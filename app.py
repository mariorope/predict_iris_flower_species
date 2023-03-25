# Importing required packages
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify, request
import pickle

# Creating the Flask application
app=Flask(__name__)

# Importing the model previous created
model = pickle.load(open("model.pkl", "rb"))

# Creating the application routes including the home and predict paths
@app.route("/")
def home():
	return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
	
	# For rendering results on HTML GUI
	features = [[float(x) for x in request.form.values()]]
	prediction = model.predict(features)

	# Result of the prediction
	output = round(prediction[0], 1)

	# Creating a text for the flower species
	if output == 0 :
		text = "This plant is from the Setosa species"
	elif output == 1:
		text = "This plant is from the Versicolor species"
	elif output == 1:
		text = "This plant is from the Virginica species"
	else:
		text = "This plant is unkown!"

	# Rendering the results of the prediction
	return render_template("index.html", prediction_text="{}.".format(text))

@app.route('/predict_api/')
def price_predict():
	model = pickle.load(open('model.pkl', 'rb'))
	sepal_length = request.args.get('sepal_length')
	sepal_width = request.args.get('sepal_width')
	petal_length = request.args.get('petal_length')
	petal_width = request.args.get('petal_width')

	test_df = pd.DataFrame({'Sepal length':[sepal_length], 'Sepal width':[sepal_width], \
		'Petal length':[petal_length], 'Petal width':[petal_width]})

	pred = model.predict(test_df)

	if pred == 0:
		pred_plant = "Setosa"
	elif pred == 1:
		pred_plant = "Versicolor"
	elif pred == 2:
		pred_plant = "Virginica"
	else:
		pred_plant = "Not known"

	return jsonify({'Plant specie': pred_plant})

# Initializing the application
if __name__ == "__main__":
	app.run(port=5000, debug=True)