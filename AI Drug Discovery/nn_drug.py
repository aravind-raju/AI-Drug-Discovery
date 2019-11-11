# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 06:44:31 2019

@author: arvin
"""

from sklearn.preprocessing import StandardScaler
from pandas import read_csv
import category_encoders as ce
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt
from functions import *
from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(2)

#opti = {'SGD': optimizers.SGD(0.001), 'RMSprop': optimizers.RMSprop(0.001),
#        'Adagrad': optimizers.Adagrad(0.001), 'Adadelta' : optimizers.Adadelta(0.001),
#        'Adamax' : optimizers.Adamax(), 'Nadam' : optimizers.Nadam()}

drop_train = read_csv('drop_train.csv')
cat_col = get_catcol()
#frequency encoding
#fe = drop_train.groupby('col3650').size()/len(drop_train)
#drop_train.loc[:, 'freq_col3650'] = drop_train.col3650.map(fe)
#mean encoding
me = ['col403', 'col985', 'col1343', 'col1509', 'col2463', 'col2895', 'col3650']
X = drop_train.copy()
y = drop_train.Score
te = ce.target_encoder.TargetEncoder(cols=me)
X = te.fit_transform(X, y))
#creating new feature
drop_train = pca(drop_train, cat_col, 29)

model_col = ['col403', 'col985', 'col1343', 'col1509', 'col2463', 'col2895',
       'col3650', 'col87', 'col118', 'col918', 'col2613', 'col3111']+ pca_col
X = X[model_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)
""""
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=50, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=5)
results = cross_val_score(pipeline, X, y, cv=kfold)

model = KerasRegressor(build_fn=baseline_model, epochs=50, verbose=0)
batch_size = list(range(100, 251, 50))
epochs = [10, 50, 100]
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
neurons = list(range(25, 60, 5))
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2')
grid_result = grid.fit(X, y)
"""

#local test
model, r2 = local_test(X_train, y_train, X_test, y_test, baseline_model)
regr = AdaBoostRegressor(base_estimator=model, random_state=0, n_estimators=100, learning_rate=0.01)
regr.fit(X, y)
#final submission
history = final_submission(X, y, baseline_model, model_col, 'test.csv')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()