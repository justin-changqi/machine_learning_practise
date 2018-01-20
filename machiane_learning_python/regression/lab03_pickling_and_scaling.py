import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
# for plot graphic
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

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
# print (forecast_out, "days")
# shift(negtive) shift down, shift(positive) shift up
df['label'] = df[forecast_col].shift(-forecast_out)

# print (df.head())

X = np.array(df.drop(['label'], 1))
# scale data to "Gaussian with zero mean and unit variance"
X = preprocessing.scale(X)
X = X[:-forecast_out]
# use to predict against
# X_lately = X[len(X-forecast_out):len(X)-1]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])
# 20% of data we want to actually use as testing data.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
# classifier
clf = LinearRegression(n_jobs=-1)
# clf = svm.SVR()
# clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)
# save training result
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
# load training result
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)
accuracy = clf.score(X_test, y_test)
# print (accuracy)
forecast_set = clf.predict(X_lately)

print (forecast_set, 'USD')
print (accuracy, '% accuracy')
print (forecast_out, 'days')

df['Forecast'] = np.nan

last_day = df.iloc[-1].name
last_unix = last_day.timestamp()
one_day = 86400 # sec in one day
next_unix = last_unix + one_day

for i in forecast_set:
    next_data = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_data] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
