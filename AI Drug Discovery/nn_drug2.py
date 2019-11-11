# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:25:08 2019

@author: arvind
"""

import category_encoders as ce
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas import read_csv
from functions import *
from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(2)

df = read_csv('drop_train.csv')
drop_col = get_catcol()
#drop_col += ['col87', 'col118', 'col918', 'col2613', 'col3111']
df.drop(drop_col, axis=1, inplace=True)

cat_col = [i for i in df.columns if len(df[i].unique()) < 21 and len(df[i].unique()) >= 5]
value_count, drop_col = {}, []
hashes, te = [], []
for i in cat_col:
    dct = df[i].value_counts()
    if next(iter(dct))/13731 <.66:
        value_count[i] = dict(df[i].value_counts())
        if len(dct) > 10:
            hashes.append(i)
        else:
            te.append(i)
    else:
        drop_col.append(i)
df.drop(drop_col, axis=1, inplace=True)
cat_col = [x for x in cat_col if x not in drop_col]

X = df.copy()
y = df.Score
X.drop('Score', axis=1, inplace=True)
leoo = ce.LeaveOneOutEncoder(cols=te)
me = ce.MEstimateEncoder(cols=hashes) 
X = leoo.fit_transform(X, y)
X = me.fit_transform(X, y)
X = pca(X, cat_col, 30)
X.drop(cat_col, axis=1, inplace=True)

rf = RandomForestRegressor()
rf.fit(X,y)
plt.plot(rf.feature_importances_)

model_col = ['col84', 'col85', 'col917', 'col2612', 'col3109', 'col3108', 'col87', 'col118', 'col918', 'col2613', 'col3111', 'col88', 'col2614', 'pca0', 'pca1']
X = df[model_col]
y = df.Score
X_train, X_test, y_train, y_test = train_test_split(X[model_col], y, train_size=0.80)
print(local_test(X_train, y_train, X_test, y_test, baseline_two))