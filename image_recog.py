import os
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from PIL import Image

def resize_and_crop(image,size):
    img_ratio = image.size[0]/ float(image.size[1])
    ratio = size[0]/float(image.size[1])
    if ratio > img_ratio:
        image = image.resize((size[0],(int(size[0]*image.size[1]/image.size[0]))),Image.ANTIALIAS)
        image = image.crop((0, 0, 30, 30))
    elif ratio < img_ratio:
        image = image.resize((size[1], (int(size[0] * image.size[1] / image.size[1]))),Image.ANTIALIAS)
        image = image.crop((0, 0, 30, 30))
    else:
        image = image.resize((size[0],size[1]),Image.ANTIALIAS)
        #image = image.crop((0, 0, 30, 30))
    return image

X=[]
y=[]

for path,subdirs,files in os.walk('/home/william/PycharmProjects/scikitlearn/Bmp/'):
    for filename in files:
        f = os.path.join(path,filename)
        #print(f)
        img = Image.open(f).convert('LA')
        img_resized = resize_and_crop(img,(30,30))
        print(img_resized.getdata())
        img_resized = np.asarray(img_resized.getdata(),dtype=np.float64, order='C')
        print(img_resized)
        img_resized.reshape((1800,1))
        target = filename[3:filename.index('-')]
        X.append(img_resized)
        y.append(target)

#X = np.array(X)
#X = X.reshape(X.shape[:2])

print(X)
print(y)
classifier = SVC(verbose=2,kernel='poly',degree=3)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)
classifier.fit(X_train,y_train)
predictions = classifier.predict(X_test)
classification_report(y_test,predictions)



