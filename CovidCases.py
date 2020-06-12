import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from datetime import date
import pickle
import io
import os
cases = pd.read_csv('Book.csv')

cases = cases[cases['Cases'].notna()]

X = cases[['Day']]
y = cases['Cases']


#Just to check how Linear regression works
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)
lm = LinearRegression()
lm.fit(X_train,y_train)
y_pred = lm.predict(X_train)
plt.plot(X_train,y_pred,color='g',label='Linear Regression')


polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(X)
poly = LinearRegression()
poly.fit(x_poly, y)
y_poly_pred = poly.predict(x_poly)
plt.scatter(X, y, color="b",s=15)
plt.plot(X,y_poly_pred,color='r',label='Polynomial Regression')


r2 = metrics.r2_score(y,y_poly_pred)
print(r2)
#Give input of date in year, month, day
def calculateDays():
    f_date = date(2020, 3, 1)
    l_date = date(2020, 6, 11) #Selected date
    delta = l_date - f_date
    return delta.days+1

def findForNthDay(n):
    x_poly = polynomial_features.fit_transform(n.reshape(-1,1))
    predicts = poly.predict(x_poly)
    return predicts


# Creating a pickle file for the classifier
filename = 'polynomial_features.pkl'
filename_model = 'ploynomail-model.pkl'
print(pickle)
pickle.dump(polynomial_features, open(filename, 'wb')) 
pickle.dump(poly, open(filename_model, 'wb')) 

def linePlot():
    casesFilter = []
    
    for (index, columnData) in cases.iterrows():
        if(index % 5 == 0 ):
            casesFilter.append(cases.iloc[index,:])
    dataframeFilter = pd.DataFrame(np.array(casesFilter),columns = ['Date', 'Day','Cases'])
    #convert Cases column type to float
    dataframeFilter.Cases = dataframeFilter.Cases.astype(float)
    #convert Date column type to datetime
    dataframeFilter.Date = dataframeFilter.Date.astype('datetime64[ns]')
    dataframeFilter['Date'] = dataframeFilter['Date'].map(lambda x:x.date())
    print('linePlot', dataframeFilter)
    plt.subplots(figsize=(11, 9))
    sns.lineplot(x="Date", y="Cases", data=dataframeFilter)
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    my_path = os.path.dirname(__file__)+"\static\images"
    print(os.path.dirname(__file__))

    my_file = 'graph.png'
    plt.savefig(os.path.join(my_path, my_file))
    return "\static\images\graph.png"
