#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint 
from keras.optimizers import SGD,RMSprop,Adam
from keras.utils import np_utils
from keras.initializers import constant
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
#import theano
import tensorflow
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.layers.advanced_activations import LeakyReLU
from glob import glob

from keras import backend as K







path1 = 'T1PatchesBitplane' 
   

listing = os.listdir(path1) 
num_samples=size(listing)



print (num_samples)

imlist = os.listdir(path1) 

#im1 = array(Image.open(path1 + '//'+ imlist[0])) 
#im2 = array(Image.open(path1 + '//'+ imlist1[0])) 
#from sklearn import preprocessing 
#immatrix = array([preprocessing.scale(array(io.imread(path1+ '//' + im1)))
#             for im1 in sorted(imlist)],'f')
immatrix = array([array(io.imread(path1+ '//' + im1).reshape(8,31,31))
             for im1 in sorted(imlist)],'f')
print ("Shape is "+ str(immatrix.shape))
label=np.ones((num_samples,), dtype = np.int)

label[0:500] = 0

label[500:1000] = 1

label[1000:1500] = 2

label[1500:2000] = 3

label[2000:] = 4




#data,Label = shuffle(immatrix,label, random_state=2)

train_data = [immatrix,label]






batch_size = 25

num_classes = 5

nb_epoch = 50





#(X_train, y_train) = (train_data[0],train_data[1])
#(X_test, y_test) = (test_data[0],test_data[1])


#print ('X.shape = ',X.shape)
#print ('y.shape = ',y.shape)


X_train, X_test, y_train, y_test = train_test_split(train_data[0], train_data[1], test_size=0.2)




X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#X_train /= 65536
#X_test /= 65536

X_train /= 255
X_test /= 255


print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

#%%
K.set_image_dim_ordering('th')

        
                
        #first set of CONV => CONV => CONV => LReLU => MAXPOOL
model = Sequential()

#First convolution Layer
model.add(Convolution2D(512, (3, 3), padding='same', input_shape=(8,31,31))) #Try for 25x25 patch
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=1))
model.add(Dropout(0.50))

model.add(Convolution2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=1))
model.add(Dropout(0.50))

model.add(Convolution2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=1))
model.add(Dropout(0.40))

model.add(Convolution2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=1))
model.add(Dropout(0.40))

model.add(Flatten()) # No dropout after flattening.
model.add(Dense(500))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
#opt = SGD(lr=0.1)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer= sgd, metrics = ['accuracy','top_k_categorical_accuracy'])

print ('Done.')
checkpoint = ModelCheckpoint('weights-Test-CNN.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)

import pandas

hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
               verbose=1,shuffle=True, validation_data=(X_test, Y_test),callbacks=[checkpoint])
pandas.DataFrame(hist.history).to_csv("SirHAFEEZ20thDecember.csv")                    

#hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#               verbose=1,shuffle=True, validation_data=(X_test, Y_test))
#pandas.DataFrame(hist.history).to_csv("14thDechistory1.csv")                    



train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(nb_epoch)


from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
  

y_pred = model.predict_classes(X_test)
print(y_pred)

p=model.predict_proba(X_test) # to predict probability

target_names = ['class 0(normal)', 'class 1(necros)', 'class 2(adema)','class 3(enhancing)','class 4(non-enhancing)']
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

# saving weights

fname = "weights-Test-CNN.hdf5"
model.save_weights(fname,overwrite=True)
