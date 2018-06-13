import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import matplotlib.cm as cm
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd

digits = fetch_mldata('MNIST-original',data_name='data',target_name='label')
counter = 1
for i in range(1, 10):
    for j in range(1,6):
        plt.subplot(9,5,counter)
        #plt.imshow(digits[(i-1)*8000+j].reshape((28,28)),cmap=cm.Greys_r)
        plt.imshow(digits['data'][(i-1) * 7000 + j].reshape(28, 28))
        plt.axis('off')
        counter +=1
#digits = np.array(digits)
#plt.show()
num_0 = digits['data'][0]
print(digits['data'])
print(digits['target'])
#__name__ == __main__ to check if running as the main script, and is not being imported
print(__name__)


if __name__ == '__main__':
    X,y = digits['data'],digits['target']
    X = X/255.0*2 -1
    X_test,X_train,y_test,y_train = train_test_split(X,y)

    pipeline = Pipeline([('clf',SVC(kernel='rbf',gamma=0.01,C=100))])
    print(pipeline.get_params().keys())
    print(X_train.shape)
    parameters = {'clf__gamma':(0.01,0.03,0.1,0.5,1),
        'clf__C':(0.1,0.3,1,3,10,30)}
    gridsearch = GridSearchCV(pipeline,parameters,scoring='accuracy',n_jobs=-1,verbose=2)
    gridsearch.fit(X_train,y_train)
    print('best score %s',gridsearch.best_score_)
    print('best params:')
    best_params = gridsearch.best_params_
    for name in sorted(parameters.keys()):
        print(name , best_params[name])
    predictions = gridsearch.predict(X_test)
    classification_report(y_test,predictions)
    print(gridsearch.predict(num_0))

