import pandas as pd
import numpy as np

print('Loading data...')
path = open("dldata/Data.csv", "r")
data = pd.read_csv(path,header=0)
data.index = pd.to_datetime(data.Index)
data.drop(data.columns[[0]], axis=1,inplace = True)

def labeler3D(x):
    if x>=0.05:
        return 2

    if x<=-0.05:
        return 0
    else:
        return 1
    
multi_step = True
n_in = n_out = 90
labeler = labeler3D

cryptos = ['btc', 'dash', 'xrp', 'xmr', 'ltc', 'doge', 'nxt', 'nmc']

#To know how these tables are created look at basic_result function
testLabeled = pd.read_pickle('dldata/testLabeled.pkl')
prices = pd.read_pickle('dldata/prices.pkl')
crix = pd.read_pickle('dldata/crix.pkl')
Results = pd.read_pickle('dldata/Results.pkl')
dataset = pd.read_pickle('dldata/dataset.pkl')
marketK = data.loc[:, list(filter(lambda x: 'market' in x, data.columns))]
marketK = marketK.loc[testLabeled.index, :]

# Caculate returns for each crypto:
crypto_returns = pd.DataFrame(prices.iloc[n_out:].values/prices.iloc[:-n_out].values - 1)
crypto_returns.index = prices.iloc[:-n_out].index
crypto_returns.columns = prices.columns


risk_free_returns = 1/100

"""Return = pd.read_pickle('C:\\Users\\BS\\Documents\\NewData\\Algorithm\\Portfolio comparison\\Return.pkl')

TotalLabeled = Return[list(
                            list(filter(lambda x: '_return(t)' in x, Return.columns)) + 
                            list(filter(lambda x: '_return(t+%s)' % n_out in x, list(filter(lambda x: '(t+' in x, Return.columns))))
                        )]
TotalLabeled = np.exp(TotalLabeled) - 1

for col in list(filter(lambda x: '_return(t+%s)' % n_out in x, TotalLabeled.columns.values)):
    TotalLabeled['class_%s' % col] = TotalLabeled.loc[:,col].apply(labeler,1)"""

def load_prediction(mypath):
    #LSTM
    path = open(mypath, "r")
    prediction = pd.read_csv(path,header=0)
    prediction.index = pd.to_datetime(prediction.Index)
    prediction.drop(prediction.columns[[0]], axis=1,inplace = True)
    return prediction

def marketK_weights(trading_signal):
    weights = (trading_signal * marketK.values).multiply(1/(trading_signal.abs() * marketK.values).sum(1), 'index')
    return weights

def price_weights(trading_signal):
    weights = trading_signal
    weights = weights*prices.iloc[:-n_out].values
    div = prices.iloc[:-n_out].abs().sum(1).values
    weights = weights.multiply(1/div, 'index')
    return weights

def equal_weights(trading_signal):
    weights = trading_signal
    div = (weights != 0).sum(axis=1).values
    weights = weights.multiply(1/div, 'index')
    return weights

def summed_returns(trading_signal):
    return np.sum(trading_signal * crypto_returns.values, 1)


def long_short_signal(short, prediction):
    trading_signal = prediction-1
    if short == False:
        trading_signal = trading_signal.replace(-1, 0)
    return trading_signal


def weighted_portfolio(short, weight):
    if short == True:
        return np.nan_to_num((weight*crypto_returns.values).sum(1)  + 2 * weight[weight < 0].fillna(0).abs().sum(1) * risk_free_returns)
    else:
        return np.nan_to_num((weight*crypto_returns.values).sum(1))

