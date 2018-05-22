import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

model = LinearRegression()

X = np.array([[6],[8],[10],[14],[18]]).reshape(-1,1)
y = [7,9,13,17.5,18]
model.fit(X,y)

test_pizza = np.array([[12]])
predicted_price = model.predict(test_pizza)[0]
print("a 12 pizza should cost $%.2f"%predicted_price)

plt.figure()
plt.title('this is a graph')
plt.xlabel('diameter')
plt.ylabel('price')
plt.plot(X,y,'k.')
plt.axis([0,25,0,25])
plt.grid(True)

print("%.2f" %np.mean((model.predict(X)-y)**2))

