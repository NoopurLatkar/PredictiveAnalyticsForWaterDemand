# -*- coding: utf-8 -*-
"""
Created on Wed May 29 18:07:58 2019

@author: hp
"""

#importing all necessary libraries
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime
import pickle
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error

#------------------------------------------------------------------------------
#Data preprocessing 

#reading data from files
dirname = 'C:/Users/hp/Desktop/BE proj/Water Consumption'
filenames = os.listdir(dirname)
filenames.sort(key=lambda date: datetime.strptime(date, '%b %Y.xls'))

pop = pd.read_excel('C:/Users/hp/Desktop/BE proj/Population.xlsx')
temp = pd.read_excel('C:/Users/hp/Desktop/BE proj/Temp.xlsx')

i=0
for fname in filenames:
    if fname.endswith(".xls"):
        #reading only necessary columns
        fname = pd.read_excel(fname,parse_cols = [2,7,8])
        #renaming column titles in dataframe
        fname.columns = ['Meter_No','Previous','Current']
        fname = fname[fname.Meter_No != 'Meter No']
        #pruning out all NAN value rows
        fname = fname.dropna(how='all')
        #dropping rows contaning strings as meter nos
        fname = fname[~fname.Meter_No.str.contains('[a-z | A-Z]', na=False)]
        #converting all meter nos to datatype int
        fname['Meter_No'] = fname['Meter_No'].astype(int)
        #sorting data on meter no values
        fname.sort_values(by=['Meter_No'], axis=0,
                          ascending=True, inplace=True) 
        fname['Consumption'] = fname['Current'] - fname['Previous'].astype(int)
        #pruning out all negative consumption rows
        fname=fname[fname['Consumption']>0]
        filenames[i] = fname
    i += 1

#------------------------------------------------------------------------------
#Wardwise monthly consumption data
    
commonlist = pd.DataFrame()
for fname1 in filenames:
    for fname2 in filenames:
        common = fname1.merge(fname2,on=['Meter_No'])
        common.drop_duplicates(subset ="Meter_No", 
                    keep = False, inplace = True) 
        fname1 = common    
commonlist=fname1

cols_of_interest = ['Meter_No','Consumption_x','Consumption_y']
commonlist = commonlist[cols_of_interest]
commonlist.columns = ['Meter_No','Dec_16','Jan_17','Feb_17','Mar_17','Apr_17','May_17','Jun_17','Jul_17','Aug_17','Sep_17','Oct_17','Nov_17','Dec_17','Jan_18','Feb_18','Mar_18','Apr_18','May_18','Jun_18','Jul_18','Aug_18','Sep_18','Oct_18','Nov_18','Dec_18','Dec_16_1']
commonlist=commonlist.drop(commonlist.columns[26], axis=1)  
    
commonlist['Ward_No'] = np.random.randint(1, 17, commonlist.shape[0])


#------------------------------------------------------------------------------
#taking input from user and filtering out data

new=pd.DataFrame()
n=pd.DataFrame()
commonmonths = commonlist.drop(['Meter_No','Ward_No'], axis=1)
columns=list(commonmonths)

number=pd.DataFrame([['Jan', 1],['Feb',2],['Mar',3],['Apr',4],['May',5],['Jun',6],['Jul',7],['Aug',8],['Sep',9],['Oct',10],['Nov',11],['Dec',12]])
number.columns=['Month','Number']

for c in columns:
    i = 0
    n['Meter_No'] = commonlist['Meter_No']
    n['Ward_No']=commonlist['Ward_No']
    firstpart = c[:len(c)//2]
    n['Mnth'] = firstpart
    tempu = temp[temp['Month'].str.contains(c)]
    t = tempu.iloc[0]['Avg. Temp.']
    n['Temperature'] = t
    n['Consumption'] = commonmonths[c]
    firstpart = c[0:len(c)//2]
    n['Population'] = "" 
    n['Month'] = "" 
    for index, row in n.iterrows():
        w = row['Ward_No'] 
        popu = pop.loc[pop['Wards'] == w]
        n.iloc[i, n.columns.get_loc('Population')] = popu.iloc[0]['Population']
        
        m = row['Mnth']
        numb = number.loc[number['Month'] == m]
        n.iloc[i, n.columns.get_loc('Month')] = numb.iloc[0]['Number']
        i += 1
    frames=[n]
    new = new.append(frames)

new = new.drop("Mnth", axis=1)
new = new.drop("Meter_No", axis=1)
new = new[['Ward_No','Month','Temperature','Population','Consumption']]

#------------------------------------------------------------------------------


label = new['Consumption']
#xxx = new[['Ward_No','Month','Temperature','Population']]

# conversion to numpy array
x, y = new.values, label.values

# scaling values for model
x_scale = MinMaxScaler()
y_scale = MinMaxScaler()

X_temp = x_scale.fit_transform(new[['Temperature']])
X_pop = x_scale.fit_transform(new[['Population']])
# =============================================================================
X_cons = x_scale.fit_transform(new[['Consumption']])
# =============================================================================

X = pd.DataFrame()
X['Ward_No']=new['Ward_No']
X['Month']=new['Month']
X['Temp']=X_temp
X['Pop']=X_pop
# =============================================================================
X['Cons']=X_cons
# =============================================================================
Y = y_scale.fit_transform(y.reshape(-1,1))

# splitting train and test
X_train, X_test, y_train, y_test = train_test_split(X.values, Y, test_size=0.33)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, 5)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)


filename = 'lstm_model.pkl'
pickle.dump(model, open(filename, 'wb'))
 
# load the model from disk
# =============================================================================
# Ward = 10
# Month = 7
# Temperature = 23
# Population = 1437
# Consumption = 0
# 
# X = pd.DataFrame()
# X = [(Ward,Month,Temperature,Population,Consumption)]
# print(X)
# X = array(X)
# X = X.reshape(X.shape[0], 5)
# X = X.reshape(X.shape[0],1,5)
# 
# yhat = model.predict(X)
# yhat = y_scale.inverse_transform(yhat)
# print(yhat)
# =============================================================================

#new[new['Ward'] == Ward]
# =============================================================================
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, y_test)
# print(result)
# =============================================================================

#------------------------------------------------------------------------------
#Testing phase

score = model.evaluate(X_test, y_test)
print('Score: {}'.format(score))
yhat = model.predict(X_test)
yhat = y_scale.inverse_transform(yhat)
print(yhat)
y_test = y_scale.inverse_transform(y_test)

plt.plot(yhat[-100:], label='Predicted')
plt.plot(y_test[-100:], label='Ground Truth')
plt.legend()
plt.show()

mean_squared_error(y_test, yhat)


