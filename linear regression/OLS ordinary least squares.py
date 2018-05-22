import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[6],[8],[10],[14],[18]]).reshape(-1,1)

y = np.array([7,12,16,17.5,20])
Xtest = np.array([[6],[8],[10],[14],[18]]).reshape(-1,1)
ytest = np.array([4,10,15,17.5,22])
y_bar = y.mean()

""" print(X.size)
x_bar = X.mean()
print(x_bar)
covariance = np.multiply((X - x_bar).transpose(), y - y_bar).sum() / (X.size-1)
print(covariance)
print("also covar %.2f" %np.cov(X.transpose(),y)[0,1])
variance =((X- x_bar)**2).sum() / (X.shape[0]-1)
print(variance)
"""
model = LinearRegression()
model = LinearRegression.fit(model, X,y)
print(LinearRegression.predict(model, 10))
print(model.score(Xtest,ytest))
