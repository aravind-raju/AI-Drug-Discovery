# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:41:59 2019

@author: arvin
"""

from pandas import read_csv
import numpy as np

train = read_csv('train.csv')
types = train.dtypes
integer = np.array(types[types==np.int64].index)
floats = np.array(types[types==np.float64].index)
boolean = []
for i in train.columns:
    if len(train[i].unique()) == 2:
        boolean.append(i)
integer = [x for x in integer if x not in boolean]

int_model, r2 = local_test(X_train, y_train, X_test, y_test, baseline_model)
bool_model, r2 = local_test(X_train, y_train, X_test, y_test, baseline_model)
float_model, r2 = local_test(X_train, y_train, X_test, y_test, baseline_model)
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

cat_col = list(set(cat_col).difference(set(drop_col)))
#PCE to reduce dimention
pce_df = train[cat_col]
from sklearn.decomposition import PCA
sc = StandardScaler()
X_std = sc.fit_transform(pce_df)
pca = PCA(n_components=10)
X_std_pca = pca.fit_transform(X_std)


X = drop_train.copy()
model_col = list(value_count.keys())+['col87', 'col118', 'col918', 'col2613', 'col3111']
y = drop_train.Score
X = X[model_col]
    
rf = RandomForestRegressor()
rf.fit(X,y)

plt.plot(rf.feature_importances_)
plt.xticks(pd.np.arange(X.shape[1]), model_col, rotation=90)

def r2Loss(y_true, y_pred):
    return r2_score(y_true, y_pred)

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss=r2Loss, optimizer='adam')
	return model

X = drop_train[model_col].iloc[:13000, :]
y = drop_train.Score[:13000]
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=10, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=5)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)
for i, j in zip(['Ridge', 'Lasso', 'LinearRegression'], [Ridge, Lasso, LinearRegression]):
    rf = j()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(i, r2_score(y_test, y_pred))


kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

rf = LinearRegression()
rf.fit(X, y)

test = pd.read_csv('test.csv')
test = sc.transform(test[['col87', 'col118', 'col918', 'col2613', 'col3111']])
pred = rf.predict(test)

submission = pd.DataFrame(data={0:test.Id, 1:pred})
