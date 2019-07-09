# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:08:54 2019

@author: mkyh8
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:16:58 2019

@author: mkyh8
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv('train.csv')
datatest= pd.read_csv('test.csv')
datatest['SalePrice'] = dataset['SalePrice'] 

Tdata=pd.concat([dataset,datatest])
Tdata = Tdata.drop(['Alley'], axis=1)
Tdata = pd.get_dummies(Tdata)
dataset1 = dataset1.dropna()


datatest= pd.read_csv('test.csv')
datatest1 = datatest.drop(['Alley'], axis=1)
datatest1 = pd.get_dummies(datatest1)

Y_y = dataset1['SalePrice']
Tdata = Tdata.drop(['SalePrice'], axis=1)


X_train = Tdata.iloc[:1460,1:].values
Y_train = Y_y.iloc[:].values

X_test = Tdata.iloc[1460:,1:].values

'''from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)'''

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values= 'NaN', strategy = 'mean' , axis =0)
X_train = imputer.fit_transform(X_train) #fit_transform requires an array not a vector (!X_train[:,1] but !X_train[:,1])
X_test = imputer.fit_transform(X_test)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 20)
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
score = r2_score(Y_test, Y_pred)

data = {'Id': datatest['Id'], 'SalePrice': Y_pred}

submission = pd.DataFrame(data)

submission.to_csv('Submission.csv', index = False)
