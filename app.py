from flask import Flask, render_template, url_for, request
from flask_material import Material
import pandas as pd 
import numpy as np 
from sklearn.externals import joblib

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    df = pd.read_csv("data/iris.csv")
    return render_template("preview.html",df_view = df)
	
@app.route('/', methods=["POST"])
def analyze():
	if request.method == 'POST':
		pl = request.form['t01']
		sl = request.form['t02']
		pw = request.form['t03']
		sw = request.form['t04']

		# Clean the data by convert from unicode to float 
		data = [pl,sl,pw,sw]
		cdata = [float(i) for i in data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(cdata).reshape(1,-1)

		# the Model ModelClassifier-version-not-work (Iris-versicolor)
		model = joblib.load('data/ModelClassifier.pkl')

		prediction = model.predict(ex1)

	return render_template('index.html', pl=pl, sl=sl, pw=pw, sw=sw, data=data, cdata=cdata, prediction=prediction)

if __name__ == '__main__':
	app.run(debug=True)