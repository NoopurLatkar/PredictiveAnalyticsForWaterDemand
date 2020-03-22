import io
import flask
import numpy as np
import keras
import pickle
from sklearn.externals import joblib
from sklearn .preprocessing import StandardScaler
from flask import Flask, Response, render_template, request, jsonify
from keras import backend as K
from scipy import misc
from tensorflow.keras.models import Sequential
from flask import jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from numpy import array
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# def get_model():
#     global model
# model = load_model('C:/Users/hp/Examples/example1/venv/app/water_demand_GRU.h5')
# model = joblib.load('C:/Users/hp/Examples/example1/venv/app/gru_model.pkl')

model = pickle.load(open('C:/Users/hp/Examples/example1/venv/app/gru_model.pkl', 'rb'))
print ("Model loaded!")
global graph
graph = tf.get_default_graph()



# print ("Loading keras model")
# get_model()

@app.route("/")
@app.route("/index")
def index():
    print ("Opening GUI")
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_predictions():
    print ("Form entered")
    if request.method == 'POST':

        print ("Values retrieved")

        ward = request.form['Ward']
        Mnth = request.form['Month']
        print('Ward : ' +ward)
        print('Month : ' +Mnth)
        
        pop = pd.DataFrame()
        temp = pd.DataFrame()
        pop = pd.read_excel('C:/Users/hp/Desktop/BE proj/Population.xlsx')
        temp = pd.read_excel('C:/Users/hp/Desktop/BE proj/Temp.xlsx')
        print(pop)
        print(temp)

        Ward = int(ward)
        print(Ward)
        popu = pop.loc[pop['Wards'] == Ward]
        Population = popu.iloc[0]['Population']
        print(Population)
        
        number=pd.DataFrame([['Jan', 1],['Feb',2],['Mar',3],['Apr',4],['May',5],['Jun',6],['Jul',7],['Aug',8],['Sep',9],['Oct',10],['Nov',11],['Dec',12]])
        number.columns=['Month','Number']
        
        numb = number.loc[number['Month'] == Mnth]
        Month = numb.iloc[0]['Number']
        print(Month)
        
        tempu = temp[temp['Month'].str.contains(Mnth)]
        Temperature = tempu['Avg. Temp.'].mean()
        print(Temperature)

        # loaded_model = pickle.load(open(filename, 'rb'))
        
        # Month = 7
        # Temperature = 23
        # Population = 1437
        Consumption = 0

        X = pd.DataFrame()
        X = [(Ward,Month,Temperature,Population,Consumption)]
        print(X)
        X = array(X)
        X = X.reshape(X.shape[0], 5)
        X = X.reshape(X.shape[0],1,5)
      
        print("Calling model")
        with graph.as_default():
             pred = model.predict(X)  
             # y_scale = MinMaxScaler()
             # pred = y_scale.inverse_transform(pred)      
        print('Prediction is : ')
        print(pred)

        # response = {
        #     'pred' : {
        #         pred[0]
        #         }
        # }

    return flask.render_template('index.html', label = pred[0]) 

if __name__ == "__main__" :
    app.run(debug=True)