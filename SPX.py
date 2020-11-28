import yfinance as yf
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
import datetime
import math

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', None)

now = datetime.datetime.now()
curryear = now.year

########## Choose starting year and trainig period length in years
startyear = 1928
trainlen = 10

########## Choose Yahoo Finance ticker symbol
tick = yf.Ticker("^GSPC")
hist = tick.history(period="max")

hist['Shifted close'] = hist.Close.shift(periods=1)
hist['Return'] = hist['Close'] / hist['Shifted close']
hist['Return lag 1'] = hist.Return.shift(periods=1)
hist['Return lag 2'] = hist.Return.shift(periods=2)
hist['Direction'] = hist.Return > 1
histtrim = hist.drop(hist.index[0:3]) #hist[hist.index > '1928-01-04']

years = range(startyear,curryear-trainlen+1)
testyears = range(startyear+trainlen,curryear+1)
passivereturns = []
activereturns = []

for y in years:
    trainstartdate = str(y) + "-01-01"
    teststartdate = str(y + trainlen) + "-01-01"
    testenddate = str(y + trainlen) + "-12-31"
    datause = histtrim[histtrim.index >= trainstartdate]
    datatrain = datause[datause.index < teststartdate]
    datatest1 = datause[datause.index >= teststartdate]
    datatest = datatest1[datatest1.index <= testenddate]
    X = datatrain[['Return lag 1', 'Return lag 2']]
    y = datatrain.Direction
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X, y)
    Xtest = datatest[['Return lag 1', 'Return lag 2']]
    datatest['preds'] = clf.predict(Xtest)
    datatest['ReturnPredDown'] = 1
    datatest['activereturn'] = np.where(datatest.preds, datatest.Return, datatest.ReturnPredDown)
    passivereturns.append(math.prod(datatest.Return))
    activereturns.append(math.prod(datatest.activereturn))

results = pd.DataFrame({'Year':testyears,
                        'Passive return':passivereturns,
                        'Active return':activereturns})
lastrets = [datatest['Return'][datatest.index[-1]],
            datatest['Return lag 1'][datatest.index[-1]]]

print(results)
print(math.prod(results['Passive return']))
print(math.prod(results['Active return']))
print(clf.predict_proba([lastrets]))
