from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron

catergories = ['rec.sport.hockey', 'rec.sport.baseball', 'rec.autos']
newgroups_train = fetch_20newsgroups(subset='train',categories=catergories,remove=('headers','footers','quotes'))
newgroups_test = fetch_20newsgroups(subset='test',categories=catergories,remove=('headers','footers','quotes'))
print(newgroups_train)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(newgroups_train.data)
X_test = vectorizer.transform(newgroups_test.data)

classifier = Perceptron(n_iter=1000, eta0= 0.01)
classifier.fit(X_train,newgroups_train.target)
predictions  = classifier.predict(X_test)
print(classification_report(newgroups_test.target,predictions))


