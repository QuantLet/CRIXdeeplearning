import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from dldata.dldata_tools import series_to_supervised_multi, dataComplete, results_rnn, save_object
#data
dataset, Return, TotalLabeled = dataComplete()

#Save object so we do not have to recreate the tables for different testing
save_object(dataset, 'dataset.pkl')
save_object(Return, 'Return.pkl')
save_object(TotalLabeled, 'TotalLabeled.pkl')


testLabeled = pd.DataFrame()
testLabeled = TotalLabeled[-365:]
cryptos = list(dataset.columns[0:8])
Results = pd.DataFrame()
prices = pd.DataFrame()
prices = dataset[cryptos+['crix']][-(365+90):]
prices = series_to_supervised_multi(prices,0,91)
#clean data
for curr in cryptos + ['crix']:
    for i in range(1,90):
        del prices['%s(t+%d)' % (curr,i)]

#results of different models
for crypto in cryptos:
    results_rnn(Return, testLabeled, crypto)
#Model diagnosis
print('Performance report for the 8 crypto models')
print('Confusion matrix')
for crypto in cryptos:
    print()
    print('RNN performance for %s_return(t+90)' % crypto)
    print()
    print('Confusion matrix')
    print(confusion_matrix(testLabeled['class_%s_return(t+90)' % crypto],testLabeled['rnn_class_%s_return(t+90)' % crypto]))
    print()
    print('Classification report')
    print(classification_report(testLabeled['class_%s_return(t+90)' % crypto],testLabeled['rnn_class_%s_return(t+90)' % crypto]))
    print()
#Accuracies are good, F1 score more constrated

rnn_class =  list(filter(lambda x: 'rnn' in x, list(filter(lambda x: 'class' in x, testLabeled.columns))))
todayRnnPrices = (testLabeled[rnn_class] * prices.iloc[:,0:8].values).sum(1)
LaterRnnPrices = (testLabeled[rnn_class] * prices.iloc[:,9:17].values).sum(1)
Results['RNN_portfolio'] = (LaterRnnPrices-todayRnnPrices)/todayRnnPrices
Results['CRIX_portfolio'] = testLabeled['crix_return(t+90)']

#Model performance
print('Quarterly returns of RNN portfolio')
print(Results['RNN_portfolio'][[90,180,270,360]])
print()

print('Quarterly returns of CRIX portfolio')
print(Results['CRIX_portfolio'][[90,180,270,360]])
print()

print("Number of quarterly portfolio that outperforms CRIX")
pd.DataFrame(Results['RNN_portfolio'] > Results['CRIX_portfolio'])[0].value_counts()
print("Percentage of quarterly protfolio that outperforms CRIX")
print('{:.1f}'.format(235*100/(365)) + ' %')

#Plots
Results.plot(title='Performance of portfolios: quarterly returns', figsize = (10,5), color = ['blue','red'])
Results.cumsum().plot(title='Performance of portfolios: cumulative quarterly returns',
                      figsize = (10,5),
                      color = ['blue','red'])


'''
dataset, Return, TotalLabeled = dataComplete()
testLabeled = pd.DataFrame()
testLabeled = TotalLabeled[-365:]
cryptos = list(dataset.columns[0:8])


Results = pd.DataFrame()
prices = pd.DataFrame()
prices = dataset[cryptos+['crix']][-(365+90):]
prices = series_to_supervised_multi(prices,0,91)
#clean data
for curr in cryptos + ['crix']:
    for i in range(1,90):
        del prices['%s(t+%d)' % (curr,i)]

#results of different models
for crypto in cryptos:
    results_rnn(Return, testLabeled, crypto)
#Model diagnosis
print('Performance report for the 8 crypto models')
print('Confusion matrix')
for crypto in cryptos:
    print()
    print('RNN performance for %s_return(t+90)' % crypto)
    print()
    print('Confusion matrix')
    print(confusion_matrix(testLabeled['class_%s_return(t+90)' % crypto],testLabeled['rnn_class_%s_return(t+90)' % crypto]))
    print()
    print('Classification report')
    print(classification_report(testLabeled['class_%s_return(t+90)' % crypto],testLabeled['rnn_class_%s_return(t+90)' % crypto]))
    print()
#Accuracies are good, F1 score more constrated

rnn_class =  list(filter(lambda x: 'rnn' in x, list(filter(lambda x: 'class' in x, testLabeled.columns))))
todayRnnPrices = (testLabeled[rnn_class] * prices.iloc[:,0:8].values).sum(1)
LaterRnnPrices = (testLabeled[rnn_class] * prices.iloc[:,9:17].values).sum(1)
Results['RNN_portfolio'] = (LaterRnnPrices-todayRnnPrices)/todayRnnPrices
Results['CRIX_portfolio'] = testLabeled['crix_return(t+90)']

#Model performance
print('Quarterly returns of RNN portfolio')
print(Results['RNN_portfolio'][[90,180,270,360]])
print()

print('Quarterly returns of CRIX portfolio')
print(Results['CRIX_portfolio'][[90,180,270,360]])
print()

print("Number of quarterly portfolio that outperforms CRIX")
pd.DataFrame(Results['RNN_portfolio'] > Results['CRIX_portfolio'])[0].value_counts()
print("Percentage of quarterly protfolio that outperforms CRIX")
print('{:.1f}'.format(235*100/(365)) + ' %')

#Plots
Results.plot(title='Performance of portfolios: quarterly returns', figsize = (10,5), color = ['blue','red'])
Results.cumsum().plot(title='Performance of portfolios: cumulative quarterly returns',
                      figsize = (10,5),
                      color = ['blue','red'])
'''