# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
from CovidCases import linePlot as linePlot
from datetime import date
import locale

# Load the Linear Regression model
polynomial_features = pickle.load(open("polynomial_features.pkl", 'rb'))
ploynomail_model = pickle.load(open("ploynomail-model.pkl", 'rb'))
print("ploy_features", polynomial_features)
print("ploynomail-model", ploynomail_model)

app = Flask(__name__)

def calculateDays(year, month, day):
    f_date = date(2020, 3, 1)
    l_date = date(year, month, day) #Selected date
    delta = l_date - f_date
    return np.array(delta.days+1)

@app.route('/')
def home():
    linePlotPath = linePlot()
    print(linePlotPath)
    return render_template('index.html', linePlotPath=linePlotPath)
    
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        date = str(request.form['date'])
        dates = date.split('-')
        noOfdays = calculateDays(int(dates[0]),int(dates[1]), int(dates[2]))
        x_poly = polynomial_features.fit_transform(noOfdays.reshape(-1,1))
        my_prediction = ploynomail_model.predict(x_poly)
        predictNumber = int(my_prediction)
        print('myprediction', my_prediction)
        fromNumber = predictNumber - 1000
        toNumber = predictNumber + 1000
        
        locale.setlocale(locale.LC_ALL, 'en_IN')
        fromNumber = locale.format('%d', fromNumber, grouping=True)
        toNumber = locale.format('%d', toNumber, grouping=True)
        return render_template('result.html', fromNumber=fromNumber, toNumber=toNumber)

if __name__ == '__main__':
	app.run(debug=True)