# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 01:54:35 2020

@author: KAVITA
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Convolution1D, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from sklearn.model_selection import train_test_split 
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping 
import sys



    
def keras_model(X_data, y_data):
    
    x_max=max(len(X_data[x]) for x in range(len(X_data)))
    x_1=x_max
    x_2=len(X_data[0][0])
    x_0=len(X_data)
    x_conv=np.zeros((x_0,x_1,x_2))
    #x_conv[0]=X_data[0]
    for l in range(len(X_data)):
        if len(X_data[l])<x_max:
           X_data[l]= np.concatenate((X_data[l], np.zeros(shape=(x_max - len(X_data[l]), x_2))))
        x_conv[l]=X_data[l]
    #print('x_conv',x_conv.shape)


    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state = 42)
    x_train=np.asarray(x_train)
    x_test=np.asarray(x_test)
    le = LabelEncoder()
    y_train=(np.asarray(y_train)).reshape([np.asarray(y_train).shape[0],1])
    y_train=to_categorical(le.fit_transform(y_train))
    print(x_train.shape)
    y_test=(np.asarray(y_test)).reshape([np.asarray(y_test).shape[0],1])
    y_test=to_categorical(le.fit_transform(y_test))
    num_labels = y_train.shape[0]
    filter_size = 2
    num_rows = x_train.shape[1]
    num_columns = x_train.shape[2]
    num_channels = 1
    num_vid=x_train.shape[0]
    classes=int(sys.argv[1])
    if classes<6:
        classes=8
    print('y_train shape=',y_train.shape,type(y_train))
 
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=2, input_shape=(None, num_columns), padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))
    
    model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(0.1))
    
    model.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(0.1))
    
    model.add(Conv1D(filters=128, kernel_size=2, padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(0.1))

    model.add(Conv1D(filters=256, kernel_size=2, padding='same',activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(0.1))
    model.add(GlobalAveragePooling1D())
    
    model.add(Dense(classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.summary()
    return model,x_train, x_test, y_train, y_test
    

    
    
    