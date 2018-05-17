import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler    
import pickle
from keras.models import load_model
from keras import backend as K


ret = lambda x,y: np.log(y/x) #Log return

def scale(data, range):
    scaler = MinMaxScaler(feature_range=range)
    scaled = scaler.fit_transform(data)
    scaled = pd.DataFrame(scaled, columns=data.columns, index=data.index)
    return scaled, scaler

def labeler(x):
    if x>=0.05:
        return 1
    else:
        return 0

def series_to_supervised_multi(data, n_in=1, n_days=1, dropnan=True):
    #n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
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

def data_input():
    path = open("Data.csv", "r")
    data = pd.read_csv(path,header=0)
    data.index = pd.to_datetime(data.Index)
    data.drop(data.columns[[0]], axis=1,inplace = True)
    crypto = list(data.columns[0:8])
    datacrypto = data.loc[:,crypto]
    endogene = data.filter(regex='Band').columns
    endogene = list(filter(lambda x: 'EuroUK' not in x, endogene))
    endogene = list(filter(lambda x: 'EuroUS' not in x, endogene))
    endogene = list(filter(lambda x: 'Euribor' not in x, endogene))
    endogene.append('crix')
    exogene=['Euribor', 'Euribor_AM', 'Euribor_1Y', 'Euribor_3M',
           'Open_price_EuroUK', 'High_price_EuroUK', 'Low_price_EuroUK',
           'Close_price_EuroUK', 'Open_price_EuroUS', 'High_price_EuroUS',
           'Low_price_EuroUS', 'Close_price_EuroUS']
    dataset = pd.concat([datacrypto, data.loc[:,endogene], data.loc[:,exogene]], axis = 1)
    return dataset


def dataComplete(n_days=91, n_in = 89):
    dataset = data_input()
    crypto = list(dataset.columns[0:8])
    Return= series_to_supervised_multi(dataset,n_in,n_days)
    
    for curr in crypto + list(['crix']):
        for i in range(n_in,0,-1):
            Return['%s_return(t-%d)' % (curr,i)] = ret(Return['%s(t-%d)' % (curr,n_in)],Return['%s(t-%d)' % (curr,i)])
        Return['%s_return(t)' % (curr)] = ret(Return['%s(t-%d)' % (curr,n_in)],Return['%s(t)' % (curr)])
        
        for i in range(n_in,0,-1):
            del Return['%s(t-%d)' % (curr,i)]
        for i in range(1,n_days):
            Return['%s_return(t+%d)' % (curr,i)] = ret(Return['%s(t)' % curr],Return['%s(t+%d)' % (curr,i)])
        for i in range(1,n_days):
            del Return['%s(t+%d)' % (curr,i)]
        del Return['%s(t)' % curr]

    TotalLabeled = Return[list(
                                list(filter(lambda x: '_return(t)' in x, Return.columns)) + 
                                list(filter(lambda x: '_return(t+90)' in x, list(filter(lambda x: '(t+' in x, Return.columns))))
                            )]
    TotalLabeled = np.exp(TotalLabeled) - 1
    for col in list(filter(lambda x: '_return(t+90)' in x, TotalLabeled.columns.values)):
        TotalLabeled['class_%s' % col] = TotalLabeled.loc[:,col].apply(labeler,1)
    TotalLabeled.columns
    return dataset, Return, TotalLabeled

def results_rnn(Return, testLabeled, var, n_days=91):
    OutputDF = Return[list(filter(lambda x: ('%s_return' % var) in x, list(filter(lambda x: '(t+' in x, Return.columns))))]
    InputDF = Return[list(filter(lambda x: '(t+' not in x, Return.columns))]
    scaled_Inputs, scaler = scale(InputDF,(-1,1))
    TotalReturn = np.exp(OutputDF) - 1
    Labeled=pd.DataFrame()
    
    for col in list(TotalReturn.columns.values):
        Labeled[col] = TotalReturn[col]
        Labeled['class_%s' % col] = Labeled.loc[:,col].apply(labeler,1)

    Targets = list(filter(lambda x: 'class' in x, Labeled.columns))
    TargetDF = Labeled[Targets]
    
    test_size = 365
    time_steps = n_days-1
    n_features = int(len(InputDF.columns)/time_steps)
    
    #train = (scaled_Inputs[:-test_size].values,TargetDF[:-test_size].values)
    val = (scaled_Inputs[-test_size:].values,TargetDF[-test_size:].values)
    
    #input_ = train[0].reshape(train[0].shape[0], time_steps, n_features)
    #target = train[1].reshape(train[1].shape[0],train[1].shape[1],1)
    val_input_ = val[0].reshape(val[0].shape[0], time_steps, n_features)
    val_target = val[1].reshape(val[1].shape[0],val[1].shape[1],1)
    
    K.clear_session()
    model = load_model('%s_model.hdf5' % var)
    print()
    print('Model summary for %s' %var)
    print()
    print(model.summary())
    print()
    print("[%s] evaluating on the whole testing set..." % var)
    (loss, accuracy) = model.evaluate(val_input_,val_target)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
    print('Predicting')
    pred = model.predict_classes(val_input_, verbose= 0)
    pred = pd.DataFrame(pred.reshape(pred.shape[0], pred.shape[1]))
    testLabeled['rnn_class_%s_return(t+90)' % var] = pred[89].values
    testLabeled['rnn_%s_return(t+90)' % var] = testLabeled['rnn_class_%s_return(t+90)' % var]*testLabeled['%s_return(t+90)' % var]
    print()
    
    return testLabeled

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
#data
#dataset, Return, TotalLabeled = dataComplete()


#save_object(dataset, '../dataset.pkl')
#save_object(Return, '../Return.pkl')
#save_object(TotalLabeled, '../TotalLabeled.pkl')


