# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 00:49:34 2020

@author: Prince
"""

#importing dataset

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#Importing the data set
dataset= pd.read_csv('Data.csv')

x=dataset.iloc[: ,:-1].values
y=dataset.iloc[: ,-1:].values

#taking care of missing data

from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(x[:, 1:3])
x[:, 1:3]=missingvalues.transform(x[:, 1:3])
#Encoding categorical dataset


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.float)
# Encoding Y data
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)

#splitting the dataset into training set and test dataset
from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)


