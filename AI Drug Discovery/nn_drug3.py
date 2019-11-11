from pandas import read_csv
import numpy as np

train = read_csv('train.csv')
drop_train = train.T.drop_duplicates()
drop_train = drop_train.T
diff_col = [x for x in train.columns if x not in drop_train.columns]
diff_types = train[diff_col].dtypes
types = train.dtypes
integer = np.array(types[types==np.int64].index)
floats = list(types[types==np.float64].index)
boolean = []
for i in train.columns:
    if len(train[i].unique()) == 2:
        boolean.append(i)
integer = [x for x in integer if x not in boolean]