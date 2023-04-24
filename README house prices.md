## PS4HouseDataExercise
## Predict House Prices using Linear Regression Model

## import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Read the data
data = pd.read_csv('train.csv')
data.shape
train = data.iloc[0:1000,:]
train.to_json('housedata.json')
train.head()


## Investigate Lot Area and plot histogram for Lot Area

## plot histogram
plt.hist(train['SalePrice'])
## select numeric columns
numeric = train.select_dtypes(include = [np.number]).columns
numeric.shape
## calculate correlation factor between LotArea and SalePrice
corr = train[numeric].corr()
corr['LotArea']['SalePrice']
cols = corr['SalePrice'].sort_values(ascending = False).index
cols = cols[1:6]
train cols.corr train[cols].corr()['LotArea']['SalePrice']
## cast to list
cols = cols.tolist()
## pick out X cols and Y = LotArea
X = train[cols]
Y = train['LotArea']
X = X.drop(['LotArea'], axis = 1)
## build Linear Regression Model
from sklearn import linear_model
model = linear_model.LinearRegression()
model = lr.fit(X,Y)
predictions = model.predict(X)
## check how good the model is
model.score(X,Y)
## plot scatter plot between LotArea and SalePrice
plt.scatter(train['LotArea'], train['SalePrice'])
plt.show()

















## Predict SalePrice using LotArea and plot histogram for SalePrice
train['LotArea'].head()
train['LotArea'].describe()
train['LotArea'].plot(kind='hist', bins=1000)
plt.show()

## Predict SalePrice using LotShape and plot histogram for SalePrice
train['LotShape'].head()
train['LotShape'].describe()
train['LotShape'].plot(kind='hist', bins=20)
plt.show()

## Predict SalePrice using LandContour and plot histogram for SalePrice
train['LandContour'].head()
train['LandContour'].describe()
train['LandContour'].plot(kind='hist', bins=20)
plt.show()

## Predict SalePrice using LotConfig and plot histogram for SalePrice
train['LotConfig'].head()
train['LotConfig'].describe()
train['LotConfig'].plot(kind='hist', bins=20)
plt.show()




