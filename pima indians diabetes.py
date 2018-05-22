
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('./diabetes.data', header =None )
y = df[8]
X = df[[0,1,2,3,4,5,6,7]]
X_train, X_test, y_train, y_test = train_test_split(X,y , stratify=y,random_state=11)

lr = LogisticRegression()
nb = GaussianNB()
lr_scores = []
nb_scores = []

train_sizes = range(10, len(X_train), 10)
for train_size in train_sizes:
    X_slice,_, y_slice,_, = train_test_split(X_train, y_train, train_size=train_size, stratify= y_train, random_state=11)
    nb.fit(X_slice, y_slice)
    nb_scores.append((nb.score(X_test, y_test)))
    lr.fit(X_slice, y_slice)
    lr_scores.append((lr.score(X_test,y_test)))

plt.plot(train_sizes, nb_scores, label = 'nb')
plt.plot(train_sizes, lr_scores, label = 'lr')
plt.title('nb and lr accuracy')
plt.xlabel("n.training instance")
plt.ylabel('accuracy')
plt.legend()
plt.show()
