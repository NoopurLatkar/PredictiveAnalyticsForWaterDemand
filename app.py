import flask
from flask import Flask, Response, render_template, request, redirect, url_for
from flask import jsonify
from sklearn.externals import joblib
import numpy as np
from scipy import misc
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from numpy import array
import tensorflow as tf
import keras
from keras.models import load_model

# # Dump the trained decision tree classifier with Pickle
# gru_pkl_filename = 'gru_classifier_20170212.pkl'
# # Open the file to save as pkl file
# gru_model_pkl = open(gru_pkl_filename, 'wb')
# pickle.dump('C:/Users/ccoew/Desktop/BE_Project', gru_model_pkl) 
# # Close the pickle instances
# gru_model_pkl.close()

app = Flask(__name__)
# load_model = joblib.load('C:/Users/hp/Examples/example1/venv/app/gru_model.pkl')



@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('page-selection.html')

@app.route('/ajaxexample')
def ajaxexample():
    return flask.render_template('ajax-example.php')


@app.route('/charts')
def charts():
    return flask.render_template('charts.html')

@app.route('/table-data-table')
def tabledatatable():
    return flask.render_template('table-data-table.html')


@app.route('/table-basic')
def tablebasic():
    return flask.render_template('table-basic.html')

@app.route('/datasets')
def datasets():
    return flask.render_template('datasets.html')

@app.route('/prediction')
def prediction():
    return flask.render_template('prediction.html')

@app.route('/waterconsumption')
def waterconsumption():
    return flask.render_template('consumption.html')

@app.route('/population')
def population():
    return flask.render_template('population.html')

@app.route('/temperature')
def temperature():
    return flask.render_template('temperature.html')

@app.route('/charts_user')
def charts_user():
    return flask.render_template('charts_user.html')

@app.route('/logout')
def logout():
    return flask.render_template('page-lockscreen.html')

@app.route('/pagelogin')
def pagelogin():
    return flask.render_template('page-login.html')


@app.route('/selec', methods=['GET', 'POST'])
def selec():
    Mod = request.form['MyRadio']
    print(Mod)
    if request.method == 'POST':
        if Mod=='one':
            return redirect(url_for('login'))
        else:
            return render_template('index_user.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('page-login.html')

@app.route('/logg', methods=['GET', 'POST'])
def logg():
    print("inside")
    u = request.form['username']
    p = request.form['password']
    if request.method == 'POST':
        if u=='admin' and p=='password':
            return redirect(url_for('first'))
        else:
            return render_template('page-login.html')

@app.route("/first", methods=['GET', 'POST'])
def first():
    return render_template('index.html')




@app.route('/predict', methods=['POST'])
def make_predictions():
    if request.method == 'POST':

        pop = pd.read_excel('C:/Users/hp/Desktop/BE proj/Population.xlsx')
        temp = pd.read_excel('C:/Users/hp/Desktop/BE proj/Temp.xlsx')
       
        Ward = request.form['exampleInputEmail1']
        Mnth = request.form['exampleSelect1']
        Mod = request.form['optionsRadios']

        print('Ward : ' +Ward)
        print('Month : ' +Mnth)
        
        convertedWard = int(Ward)
        
        number=pd.DataFrame([['Jan', 1],['Feb',2],['Mar',3],['Apr',4],['May',5],['Jun',6],['Jul',7],['Aug',8],['Sep',9],['Oct',10],['Nov',11],['Dec',12]])
        number.columns=['Month','Number']
        
        numb = number.loc[number['Month'] == Mnth]
        Month = numb.iloc[0]['Number']
        print(Mnth)

        popu = pop.loc[pop['Wards'] == convertedWard]
        print(popu)
        Population = popu.iloc[0]['Population']
        print(Population)
        
        tempu = temp[temp['Month'].str.contains(Mnth)]
        Temperature = tempu['Avg. Temp.'].mean()
        print(Temperature)

        Consumption = 268.07;


        if Mod=='option1':
            model = pickle.load(open('C:/Users/hp/Examples/example1/venv/app/gru_model.pkl', 'rb'))
        else:
            model = pickle.load(open('C:/Users/hp/Examples/example1/venv/app/lstm_model.pkl', 'rb'))

        X = pd.DataFrame()
        X = [[Ward,Month,Temperature,Population,Consumption]]
        print(X)
        X = array(X)
        X = X.reshape(X.shape[0], 5)
        X = X.reshape(X.shape[0],1,5)

        print(X)

        pred = model.predict(X)

        max = 2000
        min = 20
         
        final_pred = (pred*(max-min)) + min
        print('Prediction is : ')
        print(final_pred)



    return flask.render_template('prediction.html', label = final_pred[0]) 

if __name__ == "__main__":
    app.run(debug=True)