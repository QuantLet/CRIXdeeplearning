#import os, sys
import pandas as pd
import pickle as pk

import matplotlib.pyplot as plt
# set style, empty figure and axes
plt.style.use('ggplot')

#sys.path.append("C:\\Users\\etien\\Documents\\Algo")

# Data:
with open("dataset.pkl", 'rb') as f:
    dataset = pk.load(f)   
with open("return.pkl", 'rb') as f:
    returns = pk.load(f)
        
train_date = pd.date_range(start = '2014-07-31', end ='2017-03-01', freq='D')
test_date = pd.date_range(start = '2017-03-02', end ='2017-10-25', freq='D')

#crix graph
crix = dataset.loc[:, 'crix']

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
x_axis = crix.index
ax.plot(x_axis, crix, color='blue', lw = 1.5)
plt.show()
plt.savefig('crix.pdf')

#Bollinger bands graph
def bollinger(close, window = 20, width = 2):
    ma = close.rolling(window = window).mean()
    sd = close.rolling(window = window).std()
    upper = ma + width*sd
    lower = ma - width*sd
    result = pd.concat([close, ma, upper, lower], axis = 1)
    result.columns = ('close', 'ma_' + str(window), 'upperband_' + str(window), 'lowerband_' + str(window))
    return result

btc = dataset['btc']
bollingerbands = bollinger(btc, window = 20, width = 2)
upper = bollingerbands['upperband_20']
lower = bollingerbands['lowerband_20']
ma = bollingerbands['ma_20']

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
# Get index values for the X axis for facebook DataFrame
x_axis = btc.index
ax.fill_between(x_axis,  upper, lower, color='lightskyblue')
# Plot Adjust Closing Price and Moving Averages
ax.plot(x_axis, btc, color='red', lw=2)
ax.plot(x_axis, ma, color='green', lw=1.5)
plt.show()
plt.savefig('btc_bb.pdf')