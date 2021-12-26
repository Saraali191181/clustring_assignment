# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 16:20:51 2021

@author: hp
"""
# importing pandas libarary 

import pandas as pd

# importing dataset
dataset = pd.read_csv('Wuzzuf_Jobs.csv')
X = dataset.iloc[:, :-1].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('YearExp', OneHotEncoder(),[5])],remainder='passthrough')
labelencoder_X = LabelEncoder()
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
x=pd.DataFrame(X)



# fatorize 
dataset['YearsExp'] = pd.factorize(dataset['YearsExp'])[0]


dataset['fact'] = pd.factorize(dataset['YearsExp'])[0]
