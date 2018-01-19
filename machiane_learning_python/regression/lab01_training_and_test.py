import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

# get Google stock data set
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
# Because ML can not work with na data
df.fillna(-99999, inplace=True)
# math.ceil(x) = smallest integer value greater than or equal to x.
# We are try to predict out 10 percent of the dataframe and you'll see that actually
# when will go out and do this.
forecast_out = int(math.ceil(0.01*len(df)))
print (forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# print (df.head())

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
# scale data to "Gaussian with zero mean and unit variance"
X = preprocessing.scale(X)

# 20% of data we want to actually use as testing data.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
# classifier
clf = LinearRegression(n_jobs=-1)
# clf = svm.SVR()
# clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print (accuracy)
