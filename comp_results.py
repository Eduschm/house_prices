#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 20:24:28 2025

@author: edu
"""
import pandas as pd 
import joblib
import numpy as np

df = pd.read_csv('data/test.csv')

df = df.drop(['PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu',
              'MasVnrType', 'Alley', 'Id'], axis=1)

print(df.info())
df = df.fillna(df.mean(numeric_only=True))
df = df.fillna('NA')

cols_to_convert = ['MSSubClass', 'YrSold', 'MoSold', 'YearBuilt']
df[cols_to_convert] = df[cols_to_convert].astype('category')

df = df.drop(['GarageCars','TotalBsmtSF','GrLivArea', 'GarageYrBlt'], axis=1)
 
df = pd.get_dummies(df, drop_first=True)

model = joblib.load(f"models/XGBRegressor_best.pkl")
y = model.predict(df)
y = np.expm1(y)

df = pd.read_csv('data/test.csv')

df = df[['Id']]
df['SalePrice'] = y
 