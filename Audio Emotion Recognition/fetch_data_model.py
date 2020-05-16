# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 03:08:30 2020

@author: KAVITA
"""
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras import regularizers
from keras.layers import Lambda
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
import numpy as np
import pickle
import os
from glob import glob
from keras.applications.vgg16 import VGG16
from model import *
from training import *
from plot_accuracy import *
from vgg19 import *



class Data_preparation(object):  
  def __init__(self,fetch_data): 
      self.fetch_data=fetch_data
    
  def get_data(self, file_list):
      def load_into(self, _filename, x, y):

          if classes==6:
              base_dir='./Data/enterface_features_encoded'
          elif classes==7:
              base_dir='./Data/savee_features_encoded'
          elif classes==8:
              base_dir='./Data/ravdess_features'
          else:
              base_dir='./Data/all_features_encoded'
        
          with open(os.path.join(base_dir,_filename), 'rb') as f:
              #print(f)
              audio_element = pickle.load(f)
              x.append(np.transpose(audio_element['audio']))
              #print(audio_element['audio'])
              y.append((audio_element['class_label']))

      x, y = [], []
      for filename in file_list:
          #print(filename)
          load_into(self,filename, x, y)
     
      return x, y
    
if __name__ == "__main__":
    
    classes=int(sys.argv[1])
    if classes==6:
        file_dir='./Data/enterface_features_encoded'
    elif classes==7:
        file_dir='./Data/savee_features_encoded'
    elif classes==8:
        file_dir='./Data/ravdess_features'
    else:
        file_dir='./Data/all_features_encoded'
        
    file_list=[]

    for files in os.listdir(file_dir):
        file_list.append(files) 

    fetch_data = Data_preparation(file_list) 
    X_data, y_data=fetch_data.get_data(file_list)
    
    model_cnn,x_train, x_test, y_train, y_test=keras_model(X_data, y_data)
    history_cnn=train_cnn(model_cnn,x_train, x_test, y_train, y_test)
    plot_history(history_cnn)
    

    #model_vgg,x_train, x_test, y_train, y_test=vgg19_model(X_data, y_data)
    #history_vgg=train_cnn(model_vgg,x_train, x_test, y_train, y_test)
    #plot_history(history_vgg)
    
    
    
    
    #python fetch_data_model.py 0

    
