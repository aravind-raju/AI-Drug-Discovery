# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:58:56 2019

@author: arvin
"""
from keras.models import Sequential
from keras import layers
from keras.wrappers.scikit_learn import KerasRegressor
from keras import optimizers
from pandas import read_csv, DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import re

def baseline_model(input_dim=41, output_dim=55, activation='relu', learning_rate=0.001, loss='mean_squared_error'):
     model = Sequential()
     model.add(layers.Dense(output_dim, input_dim=input_dim, kernel_initializer='he_normal', activation=activation))
     model.add(layers.BatchNormalization())
     model.add(layers.Dense(20, kernel_initializer='he_normal', activation=activation))
     model.add(layers.BatchNormalization())
     model.add(layers.Dense(1, kernel_initializer='he_normal'))
     optimizer = optimizers.RMSprop(learning_rate)
     model.compile(loss=loss, optimizer=optimizer)
     return model

def baseline_two(input_dim=15, output_dim=30, activation='relu', learning_rate=0.001, loss='mean_squared_error'):
     model = Sequential()
     model.add(layers.Dense(output_dim, input_dim=input_dim, kernel_initializer='he_normal', activation=activation))
     model.add(layers.BatchNormalization())
     model.add(layers.Dense(20, kernel_initializer='he_normal', activation=activation))
     model.add(layers.BatchNormalization())
     model.add(layers.Dense(1, kernel_initializer='he_normal'))
     optimizer = optimizers.RMSprop(learning_rate)
     model.compile(loss=loss, optimizer=optimizer)
     return model

def local_test(X_train, y_train, X_test, y_test, model, epoch=50, batch_size=100):
    reg = KerasRegressor(build_fn=model, epochs=epoch, batch_size=batch_size, verbose=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    reg.fit(X_train, y_train)
    X_test = sc.transform(X_test)
    y_pred = reg.predict(X_test)
    return reg, r2_score(y_test, y_pred)

def final_submission(X, y, model, model_col, test_file, epoch=50, batch_size=100):
    test = read_csv(test_file)
    reg = KerasRegressor(build_fn=model, epochs=epoch, batch_size=batch_size, verbose=0)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    history = reg.fit(X, y)
    X_test = sc.transform(test[model_col])
    y_pred = model.predict(X_test)
    submission = DataFrame(data={0:test.Id, 1:y_pred})
    submission.to_csv('submission.csv', index=False, header=False)
    return history

def pca(df, columns, output):
    pca_col = ['pca'+str(i) for i in range(output)]
    df = df.reindex(columns=df.columns.tolist() + pca_col)
    sc = StandardScaler()
    X_std = sc.fit_transform(df[columns])
    pca = PCA(n_components=output)
    df[pca_col] = pca.fit_transform(X_std)
    return df

def getDuplicateColumns(df, columns):
    #re.findall(r'\d+', 'hello 42 I\'m a 32 string 30')
    duplicateColumnNames = []
    for x in range(df.shape[1]):
        col = df.iloc[:, x]
        for y in range(x + 1, df.shape[1]):
            otherCol = df.iloc[:, y]
            if col.equals(otherCol):
                duplicateColumnNames.append(df.columns.values[y])
    return duplicateColumnNames
    
def get_catcol(cardinality=5):
    train = read_csv('train.csv')
    drop_train = train.T.drop_duplicates()
    drop_train = drop_train.T
    cat_col = []
    for i in drop_train.columns:
        if len(drop_train[i].unique()) < cardinality:
            cat_col.append(i)
    value_count = {}
    for i in cat_col:
        value_count[i] = dict(drop_train[i].value_counts())
    nunique = drop_train.nunique(axis=1) == 1
    drop_train.drop(nunique[nunique == True].index.values, axis=1, inplace=True)
    drop_col = []
    for i in value_count.keys():
        key = list(value_count[i].keys())[0]
        if value_count[i][key]/13731 >0.80:
            drop_col.append(i)
    drop_train.drop(drop_col, axis=1, inplace=True)
    for k in drop_col:
        value_count.pop(k, None)
    return list(set(cat_col).difference(set(drop_col)))
    

