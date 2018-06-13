from os import walk, path
import numpy as np
import mahotas as mh
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X = []
y = []
i = 1
for dir_path, dir_names, file_names in walk('/home/william/PycharmProjects/scikitlearn/images'):
    for fn in file_names:
            if fn[-3:] == 'pgm':
                image_filename = path.join(dir_path,fn)
                X.append(scale(mh.imread(image_filename,as_grey=True).reshape(10304).astype('float32')))
                y.append(dir_path)
    i += 1


X = np.array(X)


X_train,X_test,y_train,y_test = train_test_split(X,y)
pca = PCA(n_components=250)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)
print('orig data: %s',X_train.shape)
print('red data: %s', X_train_reduced.shape)
classifier = LogisticRegression()
classifier.fit(X_train_reduced,y_train)
predictions = classifier.predict(X_test_reduced)
print('cross val score%' ,cross_val_score(classifier,X_test_reduced,y_test))

print(classification_report(y_test,predictions))

