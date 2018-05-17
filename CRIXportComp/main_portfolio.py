#sys.path.append('C:\\Users\\BS\\Documents\\NewData\\Algorithm\\Portfolio comparison\\')
from portfolio_tools import *
import os
import pandas as pd
import matplotlib.pyplot as plt

#plt.style.use('ggplot')

print('Loading data...')
path = open("dldata/Data.csv", "r")
data = pd.read_csv(path,header=0)
data.index = pd.to_datetime(data.Index)
data.drop(data.columns[[0]], axis=1,inplace = True)
cryptos = ['btc', 'dash', 'xrp', 'xmr', 'ltc', 'doge', 'nxt', 'nmc']

#To know how these tables are created look at basic_result function

testLabeled = pd.read_pickle('dldata/testLabeled.pkl')
prices = pd.read_pickle('dldata/prices.pkl')
dataset = pd.read_pickle('dldata/dataset.pkl')
marketK = data.loc[:, list(filter(lambda x: 'market' in x, data.columns))]
marketK = marketK.loc[testLabeled.index, :]

#####################
# Quarterly returns #
#####################

crypto_returns = prices/prices.shift(90) - 1
crypto_returns = crypto_returns.dropna()

#plt.style.use('classic')

#######################
#Plot quarterly returns

x = crypto_returns.index
f, axarr = plt.subplots(4, 2, figsize = (20, 15))
axarr[0, 0].plot(x, crypto_returns['btc'], color='blue', lw = 1.5)
axarr[0, 0].set_title('btc')

axarr[0, 1].plot(x, crypto_returns['dash'], color='blue', lw = 1.5)
axarr[0, 1].set_title('dash')

axarr[1, 0].plot(x, crypto_returns['xmr'], color='blue', lw = 1.5)
axarr[1, 0].set_title('xmr')

axarr[1, 1].plot(x, crypto_returns['ltc'], color='blue', lw = 1.5)
axarr[1, 1].set_title('ltc')

axarr[2, 0].plot(x, crypto_returns['nxt'], color='blue', lw = 1.5)
axarr[2, 0].set_title('nxt')

axarr[2, 1].plot(x, crypto_returns['xrp'], color='blue', lw = 1.5)
axarr[2, 1].set_title('xrp')

axarr[3, 0].plot(x, crypto_returns['doge'], color='blue', lw = 1.5)
axarr[3, 0].set_title('doge')

axarr[3, 1].plot(x, crypto_returns['nmc'], color='blue', lw = 1.5)
axarr[3, 1].set_title('nmc')

plt.savefig('Returns_cryptos.png')
#########################


#Crix
test_size = round(dataset.shape[0]*0.2)
crix = dataset['crix'][-(test_size+n_out):]
crix_portfolio = pd.DataFrame()
crix_portfolio['CRIX_return'] = (crix.iloc[n_out:].values - crix.iloc[:-n_out].values)/crix.iloc[:-n_out].values
crix_portfolio.index = prices.iloc[:-n_out].index
crix_portfolio.plot(figsize = (10, 5), color='blue', lw = 1.5)

###################
# load prediction

#Baseline
#LSTM
path = os.getcwd() + "\\dldata\\baseline_prediction\\lstm_predictions.csv"   
base_lstm_prediction = load_prediction(path)

#RNN
path = os.getcwd() + "\\dldata\\baseline_prediction\\rnn_predictions.csv"   
base_rnn_prediction = load_prediction(path)

#TDNN
path = os.getcwd() + "\\dldata\\baseline_prediction\\mlp_predictions.csv"   
base_mlp_prediction = load_prediction(path)

#Final models
#LSTM
path = os.getcwd() + "\\dldata\\final_prediction\\lstm_final_predictions.csv"   
lstm_prediction = load_prediction(path)

#RNN
path = os.getcwd() + "\\dldata\\final_prediction\\rnn_final_predictions.csv"   
rnn_prediction = load_prediction(path)

# MLP
path = os.getcwd() + "\\dldata\\final_prediction\\mlp_final_predictions.csv"   
mlp_prediction = load_prediction(path)



################################################
########### Marketcap weighted #################
################################################
################# Long portfolio

short = False
#True portfolio
action = testLabeled.loc[:, list(filter(lambda x: 'crix' not in x,
                                        list(filter(lambda x: 'class' in x, testLabeled.columns))))].copy()
action.columns = [x + '_action' for x in list(x.split("_",3)[1] for x in action.columns.tolist())]
action = action - 1
action = action[sorted(action.columns)]
model_type = 'True'

weights = marketK_weights(action)
true_portfolio = weighted_portfolio(short, weights)

#Baseline
#LSTM
action = long_short_signal(short, base_lstm_prediction)
weights = marketK_weights(action)
base_lstm_portfolio = weighted_portfolio(short, weights)

#RNN
action = long_short_signal(short, base_rnn_prediction)
weights = marketK_weights(action)
base_rnn_portfolio = weighted_portfolio(short, weights)

# MLP
action = long_short_signal(short, base_mlp_prediction)
weights = marketK_weights(action)
base_mlp_portfolio = weighted_portfolio(short, weights)


base_long_marketK = pd.DataFrame()
base_long_marketK['crix_portfolio'] = crix_portfolio['CRIX_return']
base_long_marketK['true_portfolio'] = true_portfolio
base_long_marketK['lstm_portfolio'] = base_lstm_portfolio
base_long_marketK['rnn_portfolio'] = base_rnn_portfolio
base_long_marketK['mlp_portfolio'] = base_mlp_portfolio

base_long_marketK.plot()

f, axarr = plt.subplots(2, sharex=True, figsize = (15, 15))
axarr[0].plot(base_long_marketK, lw = 1.5)
axarr[1].plot(base_long_marketK.cumsum(), lw = 1.5)
plt.savefig('base_long_marketK_weighted.png')

###############
#Final models
#LSTM
action = long_short_signal(short, lstm_prediction)
weights = marketK_weights(action)
lstm_portfolio = weighted_portfolio(short, weights)

#RNN
action = long_short_signal(short, rnn_prediction)
weights = marketK_weights(action)
rnn_portfolio = weighted_portfolio(short, weights)

# MLP
action = long_short_signal(short, mlp_prediction)
weights = marketK_weights(action)
mlp_portfolio = weighted_portfolio(short, weights)

long_marketK = pd.DataFrame()
long_marketK['crix_portfolio'] = crix_portfolio['CRIX_return']
long_marketK['true_portfolio'] = true_portfolio
long_marketK['lstm_portfolio'] = lstm_portfolio
long_marketK['rnn_portfolio'] = rnn_portfolio
long_marketK['mlp_portfolio'] = mlp_portfolio

f, axarr = plt.subplots(2, sharex=True, figsize = (15, 15))
axarr[0].plot(long_marketK, lw = 1.5)
axarr[1].plot(long_marketK.cumsum(), lw = 1.5)
plt.savefig('long_marketK_weighted.png')

################## long-short portfolio

short = True

#True portfolio
action = testLabeled.loc[:, list(filter(lambda x: 'crix' not in x,
                                        list(filter(lambda x: 'class' in x, testLabeled.columns))))].copy()
action.columns = [x + '_action' for x in list(x.split("_",3)[1] for x in action.columns.tolist())]
action = action - 1
action = action[sorted(action.columns)]
model_type = 'True'

weights = marketK_weights(action)
true_portfolio = weighted_portfolio(short, weights)

###############
# Baseline
#LSTM
action = long_short_signal(short, base_lstm_prediction)
weights = marketK_weights(action)
base_lstm_portfolio = weighted_portfolio(short, weights)

#RNN
action = long_short_signal(short, base_rnn_prediction)
weights = marketK_weights(action)
base_rnn_portfolio = weighted_portfolio(short, weights)

# MLP
action = long_short_signal(short, base_mlp_prediction)
weights = marketK_weights(action)
base_mlp_portfolio = weighted_portfolio(short, weights)

base_long_short_marketK = pd.DataFrame()
base_long_short_marketK['crix_portfolio'] = crix_portfolio['CRIX_return']
base_long_short_marketK['true_portfolio'] = true_portfolio
base_long_short_marketK['lstm_portfolio'] = base_lstm_portfolio
base_long_short_marketK['rnn_portfolio'] = base_rnn_portfolio
base_long_short_marketK['mlp_portfolio'] = base_mlp_portfolio


f, axarr = plt.subplots(2, sharex=True, figsize = (15, 15))
axarr[0].plot(base_long_short_marketK, lw = 1.5)
axarr[1].plot(base_long_short_marketK.cumsum(), lw = 1.5)
plt.savefig('base_long_short_marketK_weighted.png')

###############
#Final models
#LSTM
action = long_short_signal(short, lstm_prediction)
weights = marketK_weights(action)
lstm_portfolio = weighted_portfolio(short, weights)

#RNN
action = long_short_signal(short, rnn_prediction)
weights = marketK_weights(action)
rnn_portfolio = weighted_portfolio(short, weights)

# MLP
action = long_short_signal(short, mlp_prediction)
weights = marketK_weights(action)
mlp_portfolio = weighted_portfolio(short, weights)

long_short_marketK = pd.DataFrame()
long_short_marketK['crix_portfolio'] = crix_portfolio['CRIX_return']
long_short_marketK['true_portfolio'] = true_portfolio
long_short_marketK['lstm_portfolio'] = lstm_portfolio
long_short_marketK['rnn_portfolio'] = rnn_portfolio
long_short_marketK['mlp_portfolio'] = mlp_portfolio

f, axarr = plt.subplots(2, sharex=True, figsize = (15, 15))
axarr[0].plot(long_short_marketK, lw = 1.5)
axarr[1].plot(long_short_marketK.cumsum(), lw = 1.5)
plt.savefig('long_short_marketK_weighted.png')

##########################################
############ Equally weighted ############
##########################################

################# Long portfolio
short = False

# True portfolio
action = testLabeled.loc[:, list(filter(lambda x: 'crix' not in x,
                                        list(filter(lambda x: 'class' in x, testLabeled.columns))))].copy()
action.columns = [x + '_action' for x in list(x.split("_",3)[1] for x in action.columns.tolist())]
action = action - 1
action = action[sorted(action.columns)]
model_type = 'True'

weights = equal_weights(action)
true_portfolio = weighted_portfolio(short, weights)

###############
# Baseline
#LSTM
action = long_short_signal(short, base_lstm_prediction)
weights = equal_weights(action)
base_lstm_portfolio = weighted_portfolio(short, weights)

#RNN
action = long_short_signal(short, base_rnn_prediction)
weights = equal_weights(action)
base_rnn_portfolio = weighted_portfolio(short, weights)

# MLP
action = long_short_signal(short, base_mlp_prediction)
weights = equal_weights(action)
base_mlp_portfolio = weighted_portfolio(short, weights)

base_long_equal = pd.DataFrame()
base_long_equal['crix_portfolio'] = crix_portfolio['CRIX_return']
base_long_equal['true_portfolio'] = true_portfolio
base_long_equal['lstm_portfolio'] = base_lstm_portfolio
base_long_equal['rnn_portfolio'] = base_rnn_portfolio
base_long_equal['mlp_portfolio'] = base_mlp_portfolio


f, axarr = plt.subplots(2, sharex=True, figsize = (15, 15))
axarr[0].plot(base_long_equal, lw = 1.5)
axarr[1].plot(base_long_equal.cumsum(), lw = 1.5)
plt.savefig('base_long_equal_weighted.png')

###############
#Final models

#LSTM
action = long_short_signal(short, lstm_prediction)
weights = equal_weights(action)
lstm_portfolio = weighted_portfolio(short, weights)

#RNN
action = long_short_signal(short, rnn_prediction)
weights = equal_weights(action)
rnn_portfolio = weighted_portfolio(short, weights)

# MLP
action = long_short_signal(short, mlp_prediction)
weights = equal_weights(action)
mlp_portfolio = weighted_portfolio(short, weights)

long_equal = pd.DataFrame()
long_equal['crix_portfolio'] = crix_portfolio['CRIX_return']
long_equal['true_portfolio'] = true_portfolio
long_equal['lstm_portfolio'] = lstm_portfolio
long_equal['rnn_portfolio'] = rnn_portfolio
long_equal['mlp_portfolio'] = mlp_portfolio

f, axarr = plt.subplots(2, sharex=True, figsize = (15, 15))
axarr[0].plot(long_equal, lw = 1.5)
axarr[1].plot(long_equal.cumsum(), lw = 1.5)
plt.savefig('long_equal_weighted.png')

################# Long-short portfolio

short = True

#True portfolio
action = testLabeled.loc[:, list(filter(lambda x: 'crix' not in x,
                                        list(filter(lambda x: 'class' in x, testLabeled.columns))))].copy()
action.columns = [x + '_action' for x in list(x.split("_",3)[1] for x in action.columns.tolist())]
action = action - 1
action = action[sorted(action.columns)]
model_type = 'True'

weights = equal_weights(action)
true_portfolio = weighted_portfolio(short, weights)

###############
# Baseline
#LSTM
action = long_short_signal(short, base_lstm_prediction)
weights = equal_weights(action)
base_lstm_portfolio = weighted_portfolio(short, weights)

#RNN
action = long_short_signal(short, base_rnn_prediction)
weights = equal_weights(action)
base_rnn_portfolio = weighted_portfolio(short, weights)

# MLP
action = long_short_signal(short, base_mlp_prediction)
weights = equal_weights(action)
base_mlp_portfolio = weighted_portfolio(short, weights)

base_long_short_equal = pd.DataFrame()
base_long_short_equal['crix_portfolio'] = crix_portfolio['CRIX_return']
base_long_short_equal['true_portfolio'] = true_portfolio
base_long_short_equal['lstm_portfolio'] = base_lstm_portfolio
base_long_short_equal['rnn_portfolio'] = base_rnn_portfolio
base_long_short_equal['mlp_portfolio'] = base_mlp_portfolio


f, axarr = plt.subplots(2, sharex=True, figsize = (15, 15))
axarr[0].plot(base_long_short_equal, lw = 1.5)
axarr[1].plot(base_long_short_equal.cumsum(), lw = 1.5)
plt.savefig('base_long_short_equal_weighted.png')

###############
# Final model

#LSTM
action = long_short_signal(short, lstm_prediction)
weights = equal_weights(action)
lstm_portfolio = weighted_portfolio(short, weights)

#RNN
action = long_short_signal(short, rnn_prediction)
weights = equal_weights(action)
rnn_portfolio = weighted_portfolio(short, weights)

# MLP
action = long_short_signal(short, mlp_prediction)
weights = equal_weights(action)
mlp_portfolio = weighted_portfolio(short, weights)

long_short_equal = pd.DataFrame()
long_short_equal['crix_portfolio'] = crix_portfolio['CRIX_return']
long_short_equal['true_portfolio'] = true_portfolio
long_short_equal['lstm_portfolio'] = lstm_portfolio
long_short_equal['rnn_portfolio'] = rnn_portfolio
long_short_equal['mlp_portfolio'] = mlp_portfolio

f, axarr = plt.subplots(2, sharex=True, figsize = (15, 15))
axarr[0].plot(long_short_equal, lw = 1.5)
axarr[1].plot(long_short_equal.cumsum(), lw = 1.5)
plt.savefig('long_short_equal_weighted.png')



########################################
############ Price weighted ############
########################################

################# Long portfolio

short = False

#True portfolio

action = testLabeled.loc[:, list(filter(lambda x: 'crix' not in x,
                                        list(filter(lambda x: 'class' in x, testLabeled.columns))))].copy()
action.columns = [x + '_action' for x in list(x.split("_",3)[1] for x in action.columns.tolist())]
action = action - 1
action = action[sorted(action.columns)]
weights = price_weights(action)
true_portfolio = weighted_portfolio(short, weights)

###############
#Baseline

#LSTM
action = long_short_signal(short, base_lstm_prediction)
weights = price_weights(action)
base_lstm_portfolio = weighted_portfolio(short, weights)

#RNN
action = long_short_signal(short, base_rnn_prediction)
weights = price_weights(action)
base_rnn_portfolio = weighted_portfolio(short, weights)

# MLP
action = long_short_signal(short, base_mlp_prediction)
weights = price_weights(action)
base_mlp_portfolio = weighted_portfolio(short, weights)

base_long_price = pd.DataFrame()
base_long_price['crix_portfolio'] = crix_portfolio['CRIX_return']
base_long_price['true_portfolio'] = true_portfolio
base_long_price['lstm_portfolio'] = base_lstm_portfolio
base_long_price['rnn_portfolio'] = base_rnn_portfolio
base_long_price['mlp_portfolio'] = base_mlp_portfolio


f, axarr = plt.subplots(2, sharex=True, figsize = (15, 15))
axarr[0].plot(base_long_price, lw = 1.5)
axarr[1].plot(base_long_price.cumsum(), lw = 1.5)
plt.savefig('base_long_price_weighted.png')

###############
#Final models

#LSTM
action = long_short_signal(short, lstm_prediction)
weights = price_weights(action)
lstm_portfolio = weighted_portfolio(short, weights)

#RNN
action = long_short_signal(short, rnn_prediction)
weights = price_weights(action)
rnn_portfolio = weighted_portfolio(short, weights)

# MLP
action = long_short_signal(short, mlp_prediction)
weights = price_weights(action)
mlp_portfolio = weighted_portfolio(short, weights)

long_price = pd.DataFrame()
long_price['crix_portfolio'] = crix_portfolio['CRIX_return']
long_price['true_portfolio'] = true_portfolio
long_price['lstm_portfolio'] = lstm_portfolio
long_price['rnn_portfolio'] = rnn_portfolio
long_price['mlp_portfolio'] = mlp_portfolio

f, axarr = plt.subplots(2, sharex=True, figsize = (15, 15))
axarr[0].plot(long_price, lw = 1.5)
axarr[1].plot(long_price.cumsum(), lw = 1.5)
plt.savefig('long_price_weighted.png')

################## Long short portfolio
short = True
###############
#True portfolio

action = testLabeled.loc[:, list(filter(lambda x: 'crix' not in x,
                                        list(filter(lambda x: 'class' in x, testLabeled.columns))))].copy()
action.columns = [x + '_action' for x in list(x.split("_",3)[1] for x in action.columns.tolist())]
action = action - 1
action = action[sorted(action.columns)]
model_type = 'True'

weights = price_weights(action)
true_portfolio = weighted_portfolio(short, weights)

###############
#Baseline

#LSTM
action = long_short_signal(short, base_lstm_prediction)
weights = price_weights(action)
base_lstm_portfolio = weighted_portfolio(short, weights)

#RNN
action = long_short_signal(short, base_rnn_prediction)
weights = price_weights(action)
base_rnn_portfolio = weighted_portfolio(short, weights)

# MLP
action = long_short_signal(short, base_mlp_prediction)
weights = price_weights(action)
base_mlp_portfolio = weighted_portfolio(short, weights)

base_long_short_price = pd.DataFrame()
base_long_short_price['crix_portfolio'] = crix_portfolio['CRIX_return']
base_long_short_price['true_portfolio'] = true_portfolio
base_long_short_price['lstm_portfolio'] = base_lstm_portfolio
base_long_short_price['rnn_portfolio'] = base_rnn_portfolio
base_long_short_price['mlp_portfolio'] = base_mlp_portfolio


f, axarr = plt.subplots(2, sharex=True, figsize = (15, 15))
axarr[0].plot(base_long_short_price, lw = 1.5)
axarr[1].plot(base_long_short_price.cumsum(), lw = 1.5)
plt.savefig('base_long_short_price_weighted.png')

###############
#Final models
#LSTM
action = long_short_signal(short, lstm_prediction)
weights = price_weights(action)
lstm_portfolio = weighted_portfolio(short, weights)

#RNN
action = long_short_signal(short, rnn_prediction)
weights = price_weights(action)
rnn_portfolio = weighted_portfolio(short, weights)

# MLP
action = long_short_signal(short, mlp_prediction)
weights = price_weights(action)
mlp_portfolio = weighted_portfolio(short, weights)

long_short_price = pd.DataFrame()
long_short_price['crix_portfolio'] = crix_portfolio['CRIX_return']
long_short_price['true_portfolio'] = true_portfolio
long_short_price['lstm_portfolio'] = lstm_portfolio
long_short_price['rnn_portfolio'] = rnn_portfolio
long_short_price['mlp_portfolio'] = mlp_portfolio

f, axarr = plt.subplots(2, sharex=True, figsize = (15, 15))
axarr[0].plot(long_short_price, lw = 1.5)
axarr[1].plot(long_short_price.cumsum(), lw = 1.5)
plt.savefig('long_short_price_weighted.png')


##########################################
######### Performance comparison #########
##########################################

columns = [['Baseline model', 'Baseline model', 'Baseline model', 'Final model', 'Final model', 'Final model'],
           ['Price', 'Marketcap', 'Equal', 'Price', 'Marketcap', 'Equal']]
columns = pd.MultiIndex.from_tuples(list(zip(*columns)), names=['model', 'portfolio'])

#Performance long strategy
long_strat_perf = pd.DataFrame(pd.concat([base_long_price.cumsum().iloc[-1,:],
                                     base_long_marketK.cumsum().iloc[-1,:],
                                     base_long_equal.cumsum().iloc[-1,:],
                                     long_price.cumsum().iloc[-1,:],
                                     long_marketK.cumsum().iloc[-1,:],
                                     long_equal.cumsum().iloc[-1,:]], 1).drop('true_portfolio').values,
                          index=['CRIX', 'LSTM', 'RNN', 'MLP'],
                          columns=columns)
print(long_strat_perf)
#long_strat_perf.to_csv('long_strat_perf.csv', float_format='%.f')

#Performance long short strategy
long_short_strat_perf = pd.DataFrame(pd.concat([base_long_short_price.cumsum().iloc[-1,:],
                                     base_long_short_marketK.cumsum().iloc[-1,:],
                                     base_long_short_equal.cumsum().iloc[-1,:],
                                     long_short_price.cumsum().iloc[-1,:],
                                     long_short_marketK.cumsum().iloc[-1,:],
                                     long_short_equal.cumsum().iloc[-1,:]], 1).drop('true_portfolio').values,
                          index=['CRIX', 'LSTM', 'RNN', 'MLP'],
                          columns=columns)
print(long_short_strat_perf)
#long_short_strat_perf.to_csv('long_short_strat_perf.csv', float_format='%.f')