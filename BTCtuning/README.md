[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **BTCtuning** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet : BTCtuning

Published in : Deep Neural Networks for Cryptocurrencies Price Prediction

Description : 'Tuning of different MLP architecture for BTC trend predictions'

Keywords : neural networks, MLP, deep learning, Bitcoin, cryptocurrency, prediction, time series, stock price forecast

Author : SPILAK Bruno

Submitted : 2018/05/19

Datafile :
- data
- Data.csv

Output :
- Best model architecture
- Validation score

```

### PYTHON Code
```python

import os
import pandas as pd
import numpy as np

wdir = os.getcwd()

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

from time import time

from keras.layers import Dense
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from sklearn.model_selection import TimeSeriesSplit


# Input parameters
var = 'btc'
n_in = 5 #look back window
n_out = 2 #forecast window
batch_size = 256
epochs = 1

def labeler2D(x):
    if x>=0:
        return 1

    else:
        return 0

def series_to_supervised_multi(data, n_in=1, n_days=1, dropnan=True):  #index corresponds to (t)
    df = data.copy()
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (col,i)) for col in cols[0].columns.values]
    for i in range(0, n_days):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % col) for col in cols[0].columns.values]
        else:
            names += [('%s(t+%d)' % (col,i)) for col in cols[0].columns.values]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def create_table(var, data, exogene):
    #log returns
    returns = np.log(data.loc[:,var].shift(-1)/data.loc[:,var])
    returns = returns.dropna(axis = 0, how='all')
    returns_crix = np.log(data.loc[:,'crix'].shift(-1)/data.loc[:,'crix'])
    returns_crix = returns_crix.dropna(axis = 0, how='all')

    dataset = pd.DataFrame()
    dataset = data.loc[:, exogene].iloc[:-1,:]
    dataset[var] = returns.values
    dataset['crix'] = returns_crix.values

    dataset['ma_14_%s' % var] = returns.rolling(window=14,center=False).mean()
    dataset['ma_14_%s' % var].iloc[:13] = returns[:13].mean()
    dataset['ma_30_%s' % var] = returns.rolling(window=30,center=False).mean()
    dataset['ma_30_%s' % var].iloc[:29] = returns[:29].mean()
    dataset['ma_90_%s' % var] = returns.rolling(window=90,center=False).mean()
    dataset['ma_90_%s' % var].iloc[:89] = returns[:89].mean()

    dataset['Upper_14_%s' % var] = dataset['ma_14_%s' % var] + 2*returns.rolling(window=14,center=False).var()
    dataset['Upper_14_%s' % var].iloc[:13] = dataset['ma_14_%s' % var] + 2*returns[:13].var()
    dataset['Upper_30_%s' % var] = dataset['ma_30_%s' % var] + 2*returns.rolling(window=30,center=False).var()
    dataset['Upper_30_%s' % var].iloc[:29] = dataset['ma_30_%s' % var] + 2*returns[:29].var()
    dataset['Upper_90_%s' % var] = dataset['ma_90_%s' % var] + 2*returns.rolling(window=90,center=False).var()
    dataset['Upper_90_%s' % var].iloc[:89] = dataset['ma_90_%s' % var] + 2*returns[:89].var()

    dataset['Lower_14_%s' % var] = dataset['ma_14_%s' % var] - 2*returns.rolling(window=14,center=False).var()
    dataset['Lower_14_%s' % var].iloc[:13] = dataset['ma_14_%s' % var] - 2*returns[:13].var()
    dataset['Lower_30_%s' % var] = dataset['ma_30_%s' % var] - 2*returns.rolling(window=30,center=False).var()
    dataset['Lower_30_%s' % var].iloc[:29] = dataset['ma_30_%s' % var] - 2*returns[:29].var()
    dataset['Lower_90_%s' % var] = dataset['ma_90_%s' % var] - 2*returns.rolling(window=90,center=False).var()
    dataset['Lower_90_%s' % var].iloc[:89] = dataset['ma_90_%s' % var] - 2*returns[:89].var()

    dataset['vol_%s' % var] = data.loc[:, 'vol_%s' % var]
    dataset['market_%s' % var] = data.loc[:, 'market_%s' % var]
    return dataset

def baseline_model():
    K.clear_session()  
    model = Sequential()
    model.add(Dense(5, input_dim=input_.shape[1], kernel_initializer='uniform', activation = 'tanh'))
    #model.add(Dense(5, kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(num_classes, activation = 'softmax'))
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics = ['accuracy']
                 )
    return model
from sklearn.model_selection import cross_val_score

def create_model_two_layers(neurons1=1, neurons2=1):
    K.clear_session() 
    model = Sequential()
    model.add(Dense(neurons1, input_dim=input_.shape[1], kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(neurons2, kernel_initializer='uniform', activation = 'tanh'))

    model.add(Dense(num_classes, activation = 'softmax'))
    model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy']
             )
    print(model.summary())
    return model

def create_model_three_layers(neurons1=1, neurons2=1, neurons3=1):
    K.clear_session()  
    model = Sequential()
    model.add(Dense(neurons1, input_dim=input_.shape[1], kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(neurons2, kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(neurons3, kernel_initializer='uniform', activation = 'tanh'))

    model.add(Dense(num_classes, activation = 'softmax'))
    model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy']
             )
    print(model.summary())
    return model

def create_model_four_layers(neurons1=1, neurons2=1, neurons3=1, neurons4=1):
    K.clear_session() 
    model = Sequential()
    model.add(Dense(neurons1, input_dim=input_.shape[1], kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(neurons2, kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(neurons3, kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(neurons4, kernel_initializer='uniform', activation = 'tanh'))

    model.add(Dense(num_classes, activation = 'softmax'))
    model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy']
             )
    print(model.summary())
    return model

def create_model_ten_layers(neurons1=1, neurons2=1, neurons3=1, neurons4=1, neurons5=1,
                            neurons6=1, neurons7=1, neurons8=1, neurons9=1, neurons10=1,
                           ):
    K.clear_session() 
    model = Sequential()
    model.add(Dense(neurons1, input_dim=input_.shape[1], kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(neurons2, kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(neurons3, kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(neurons4, kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(neurons5, kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(neurons6, kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(neurons7, kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(neurons8, kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(neurons9, kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(neurons10, kernel_initializer='uniform', activation = 'tanh'))

    model.add(Dense(num_classes, activation = 'softmax'))
    model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy']
             )
    print(model.summary())
    return model

#############################################
################## Data #####################
#############################################
path = open(wdir + "/Data.csv", "r")
data = pd.read_csv(path,header=0)
data.index = pd.to_datetime(data.Index)
data.drop(data.columns[[0]], axis=1,inplace = True)

exogene=['EURIBOR_1Y', 'EURIBOR_6M', 'EURIBOR_3M',
           'Open_price_EuroUK', 'High_price_EuroUK', 'Low_price_EuroUK',
           'Close_price_EuroUK', 'Open_price_EuroUS', 'High_price_EuroUS',
           'Low_price_EuroUS', 'Close_price_EuroUS', 'Open_price_USJPY', 'High_price_USJPY', 'Low_price_USJPY',
           'Close_price_USJPY']
crix = ['crix']

dataset = create_table(var, data, exogene)
labeler = labeler2D
model_exogene=['crix', 'EURIBOR_1Y', 'EURIBOR_6M', 'EURIBOR_3M', 'Close_price_EuroUK', 'Close_price_EuroUS', 'Close_price_USJPY']
model_endogene = list(filter(lambda x: 'market' not in x,
                             list(filter(lambda x: 'vol' not in x,
                                 list(filter(lambda x: '_btc' in x, dataset.columns))))))

model_dataset = pd.DataFrame()
model_dataset = dataset.loc[:, [var] + model_endogene + model_exogene]
train_size = int(model_dataset.shape[0]*0.8)
train = model_dataset.iloc[:train_size, :]
val = model_dataset.iloc[train_size:, :]
  
#Preprocessing    
scaler = StandardScaler()
scaler.fit(train)
data_scaled = pd.DataFrame(scaler.transform(train), index = train.index, columns = train.columns)

data_sup = series_to_supervised_multi(data_scaled.loc[:,['btc']], n_in-1, n_out+1)

model_data = pd.concat([data_sup, data_scaled.iloc[(n_in-1):-n_out,:].loc[:, data_scaled.columns != 'btc']], axis = 1)

outputs = list(filter(lambda x: '(t+' in x, model_data.columns))
inputs = list(filter(lambda x: '(t+' not in x, model_data.columns))

input_ = model_data.loc[:,inputs].values
                           
target = model_data.loc[:,outputs]

#unscaling
target = ((target*scaler.scale_[0])+scaler.mean_[0]).values
target = pd.DataFrame([item.sum() for item in target])

target = np.apply_along_axis(labeler, 1, target)
num_classes = len(np.unique(target))

#Baseline crossvalidation
estimator = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=32, verbose=1)
kfold = TimeSeriesSplit(n_splits=5)
base_results = cross_val_score(estimator, input_, target, cv=kfold)
print("Results: %.2f (%.2f) Accuracy" % (base_results.mean(), base_results.std()))

# Two layers
print('Creation model')
model = KerasClassifier(build_fn=create_model_two_layers, epochs=epochs, batch_size=batch_size, verbose=1)
# define the grid search parameters
neurons1 = [5, 10, 15, 20, 25, 50, 100]
neurons2 = [5, 10, 15, 20, 25, 50, 100]
param_grid = dict(neurons1=neurons1, neurons2 = neurons2)

#Define CV
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, verbose=2, cv = TimeSeriesSplit(n_splits=5).split(input_, target))
grid_result = grid.fit(input_, target)
print('End gridsearch')

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



#Three layers
print('Creation model')
model2 = KerasClassifier(build_fn=create_model_three_layers, epochs=epochs, batch_size=batch_size, verbose=1)
# define the grid search parameters

neurons1 = [5, 10, 15, 20, 25, 50]
neurons2 = neurons1
neurons3 = neurons2
param_grid = dict(neurons1=neurons1, neurons2 = neurons2, neurons3 = neurons3)


#Define CV
t1 = time()
grid2= GridSearchCV(estimator=model2, param_grid=param_grid, n_jobs=1, verbose=1, cv = TimeSeriesSplit(n_splits=5).split(input_, target))
grid_result2 = grid2.fit(input_, target)
t2 = time()
print('End gridsearch, time: ', str(t2- t1))

# summarize results
print("Best: %f using %s" % (grid_result2.best_score_, grid_result2.best_params_))
means2 = grid_result2.cv_results_['mean_test_score']
stds2 = grid_result2.cv_results_['std_test_score']
params2 = grid_result2.cv_results_['params']
for mean, stdev, param in zip(means2, stds2, params2):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Four layers
print('Creation model')
model3 = KerasClassifier(build_fn=create_model_four_layers, epochs=epochs, batch_size=batch_size, verbose=1)
# define the grid search parameters
neurons1 = [25]
neurons2 = [10]
neurons3 = [25]
neurons4 = [5, 8, 10, 12, 15, 18, 20, 25, 50]
param_grid = dict(neurons1=neurons1, neurons2 = neurons2, neurons3 = neurons3, neurons4 = neurons4)

#Define CV
grid3= GridSearchCV(estimator=model3, param_grid=param_grid, n_jobs=1, verbose=1, cv = TimeSeriesSplit(n_splits=5).split(input_, target))
grid_result3 = grid3.fit(input_, target)
print('End gridsearch')

# summarize results
print("Best: %f using %s" % (grid_result3.best_score_, grid_result3.best_params_))
means3 = grid_result3.cv_results_['mean_test_score']
stds3 = grid_result3.cv_results_['std_test_score']
params3 = grid_result3.cv_results_['params']
for mean, stdev, param in zip(means3, stds3, params3):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
#0.555128 using {'neurons1': 50, 'neurons2': 15, 'neurons3': 15, 'neurons4': 15}

# 10 layers
print('Creation model')
model4 = KerasClassifier(build_fn=create_model_ten_layers, epochs=epochs, batch_size=batch_size, verbose=1)
# define the grid search parameters
neurons1 = [50]
neurons2 = [15]
neurons3 = [15]
neurons4 = [15]
neurons5 = [5]
neurons6 = [5]
neurons7 = [5]
neurons8 = [5]
neurons9 = [5]
neurons10 = [5]

param_grid = dict(neurons1=neurons1, neurons2 = neurons2, neurons3 = neurons3, neurons4 = neurons4,
                  neurons5=neurons5, neurons6 = neurons5, neurons7 = neurons5, neurons8 = neurons5,
                  neurons9 = neurons5, neurons10 = neurons5
                 
                 )


#Define CV
grid4= GridSearchCV(estimator=model4, param_grid=param_grid, n_jobs=1, verbose=1, cv = TimeSeriesSplit(n_splits=5).split(input_, target))
grid_result4 = grid4.fit(input_, target)
print('End gridsearch')

# summarize results
print("Best: %f using %s" % (grid_result4.best_score_, grid_result4.best_params_))
means4 = grid_result4.cv_results_['mean_test_score']
stds4 = grid_result4.cv_results_['std_test_score']
params4 = grid_result4.cv_results_['params']
for mean, stdev, param in zip(means4, stds4, params4):
    print("%f (%f) with: %r" % (mean, stdev, param))


print("Baseline model cross val score: %.2f (%.2f) Accuracy" % (base_results.mean(), base_results.std()))
print('Two layers: epochs: ', epochs, 'accuracy: ', grid_result.best_score_, grid_result.best_params_)
print('Three layers: epochs: ', epochs, 'accuracy: ',  grid_result2.best_score_, grid_result2.best_params_)
print('Four layers: epochs: ', epochs, 'accuracy: ',  grid_result3.best_score_, grid_result3.best_params_)
print('Ten layers: epochs: ', epochs, 'accuracy: ', grid_result4.best_score_, grid_result4.best_params_)

'''
Select best parameters from grid search
'''
```

automatically created on 2018-05-28