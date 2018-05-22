import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier

X_train = np.array([
    [158, 64],
    [170, 86],
    [183, 84],
    [191, 80],
    [155, 49],
    [163, 59],
    [180, 67],
    [158, 54],
    [170, 67]
])
x= np.array([[155,70]])
distances = np.sqrt(np.sum((X_train-x)**2, axis = 1))
#print(distances)
y_train = ['male', 'male', 'male', 'male', 'female', 'female', 'female', 'female', 'female']
nearest_neighbor_indices = distances.argsort()[:3]
nearest_neighbor_indices = np.take(y_train, nearest_neighbor_indices)
#print(nearest_neighbor_indices)

b = Counter(np.take(y_train, distances.argsort()[:3]))
#print(b.most_common(1)[0][0])

lb = LabelBinarizer()
y_train_binarized = lb.fit_transform(y_train)
print(y_train_binarized)
K = 3
clf = KNeighborsClassifier(n_neighbors=K)
clf.fit(X_train,y_train_binarized.reshape(-1))
prediction_binarized = clf.predict(np.array([155,70]).reshape(1,-1))[0]
print(prediction_binarized`)
predicted_label = lb.inverse_transform(prediction_binarized)
print(predicted_label
      )



plt.figure()
plt.title('Human Heights and Weights by Sex')
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')

"""
for i,x in enumerate(X_train):
    plt.scatter(x[0], x[1],c='k', marker='x' if y_train[i] == 'm' else 'D')
    plt.grid(True)
    plt.show()
"""
for i, x in enumerate(X_train):
# Use 'x' markers for instances that are male and diamond markers for instances that are female
    plt.scatter(x[0], x[1], c='k', marker='x' if y_train[i] == 'male' else 'D')
plt.grid(True)
#plt.show()