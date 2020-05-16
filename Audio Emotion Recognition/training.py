# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 03:08:30 2020

@author: KAVITA
"""
#import openSmile
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping 
import keras.backend as K
from keras import regularizers
from keras.layers import Lambda
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
import numpy as np
from plot_accuracy import *
import sys


def train_cnn(model,x_train, x_test, y_train, y_test):
    classes=int(sys.argv[1])
    if classes==6:
        emotions = ['anger', 'disgust','fear', 'happiness', 'sadness', 'surprise']
        model_name='model_enterface.h5'
    elif classes==7:
        emotions = ['anger', 'disgust','fear', 'happiness', 'sadness', 'surprise','neutral']
        model_name='model_savee.h5'
        
    elif classes==8:
        emotions = ['anger', 'disgust','fear', 'happiness', 'sadness', 'surprise','neutral','calm']
        model_name='model_ravdess.h5'
    else:
        emotions = ['anger', 'disgust','fear', 'happiness', 'sadness', 'surprise']
        model_name='model_all.h5'
        classes=6
        
    num_epochs = 100
    num_batch_size = 200
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, verbose=1, min_lr=1e-8)
    early_stop = EarlyStopping(monitor='val_acc', verbose=1, patience=20,  restore_best_weights=True)
    print('Training started')
    history = model.fit(x_train, y_train,batch_size=num_batch_size, epochs=num_epochs,validation_data=[x_test, y_test],verbose = 1,callbacks=[reduce_lr,early_stop])
    model.save(model_name)
    y_pred=model.predict_classes(x_test)
    rounded_labels2=np.argmax(y_test, axis=1)
    plot_confusionMatrix(rounded_labels2,y_pred)
    
    
    return history
   
