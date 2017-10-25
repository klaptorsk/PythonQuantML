import pandas_datareader as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mtdates
import datetime as dt
import sklearn as skl
import sklearn.datasets as datasetter
 

Start = dt.date.today() - dt.timedelta(900)

End = dt.date.today()

print(Start, End)

Portfolio = ['FB', 'LOXO', 'TEVA', 'FSLR', 'AAPL', 'AMD', 'NVDA']

PortCodes = list(range(1, len(Portfolio)+1))

df = pd.DataFrame(Portfolio, PortCodes)

df

PortAndCodes = [Portfolio], [PortCodes]

PortAndCodes[1:]

StockData = web.DataReader(Portfolio, start=End, end=Start, data_source='google').to_frame().reset_index()


##UpDown
StockData['UpDown'] = np.where(StockData['Close']>StockData['Open'],'Up', 'Down')

##Prepping Arrays for Bundle

ArrayOHLC = np.array(StockData.columns.values)

ArrayOHLC[0] = ['Index']

ArrayOHLC

ArrayTickers = np.array(StockData.minor)

ArrayTickers

UpDown = np.array(StockData.UpDown)

ArrayDates = np.array(StockData.Date)

ArrayData = np.array(StockData)



##Bundeling Dataset
StockDataSet = skl.datasets.base.Bunch(data=ArrayData, target= UpDown, features=ArrayOHLC)


##Rearranging Lables and dates to numbers for sklearn
StockDataSet.data[:,0] = mtdates.date2num(StockDataSet.data[:,0])

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

## Relabeling Tickers
le.fit(StockDataSet.data[:,1]) ## Fitting LabelEncoder

StockDataSet.data[:,1] = le.transform(StockDataSet.data[:,1] )

le.fit(StockDataSet.data[:,7])

StockDataSet.data[:,7] = le.transform(StockDataSet.data[:,7])


## Fitting to KNN model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5,weights='uniform')

X, y = StockDataSet.data, StockDataSet.target

knn.fit(X,y)



pred_x = knn.predict(X)

print(pred_x)

propa = knn.predict_proba(X)

print(propa)
