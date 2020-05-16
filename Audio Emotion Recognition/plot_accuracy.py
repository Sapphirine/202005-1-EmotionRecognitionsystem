# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:13:53 2020

@author: KAVITA
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from datetime import datetime
import sys
import tensorflow as tf
import sklearn
#from tf.math import confusion_matrix
import pickle as pkl
#import load_pickle
import numpy as np
def plot_history(history):
    dir_path='./plots/'
    now = datetime.now()
    dt_string = now.strftime("%H:%M")
    classes=int(sys.argv[1])
    plt.figure(0)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    if classes==6:
        file_name='loss_enterface.png'
        plt.savefig(dir_path+file_name)
    elif classes==7:
        file_name='loss_savee.png'
        plt.savefig(dir_path+file_name)  
    
    elif classes==8:
        file_name='loss_ravdess.png'
        plt.savefig(dir_path+file_name) 
    
    else:
        file_name='loss_all.png'
        plt.savefig(dir_path+file_name) 
        
    
    plt.figure(1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    if classes==6:
        plt.savefig(dir_path+'accuracy_enterface.png')
    elif classes==7:
        plt.savefig(dir_path+'accuracy_savee.png')
    elif classes==8:
        file_name=str(dt_string)+'accuracy_ravdess.png'
        plt.savefig(dir_path+file_name) 
    else:
        plt.savefig(dir_path+'accuracy_all.png')
    #plt.savefig(dir_path+'accuracy_savee.png')
    
    
def plot_confusionMatrix(y_true,y_pred):
    dir_path='./plots/'
    classes=int(sys.argv[1])
    classes=8
    now = datetime.now()
    dt_string = str(now.strftime("%H:%M"))
    if classes==6:
        emotions = ['anger', 'disgust','fear', 'happiness', 'sadness', 'surprise']
    elif classes==7:
        emotions = ['anger', 'disgust','fear', 'happiness', 'sadness', 'surprise','neutral']
    elif classes==8:
        emotions = ['anger', 'disgust','fear', 'happiness', 'sadness', 'surprise','neutral','calm']
    else:
        emotions = ['anger', 'disgust','fear', 'happiness', 'sadness', 'surprise','neutral','calm']
        classes=8
    
    tf.compat.v1.disable_eager_execution()
    sess=tf.compat.v1.Session()
    cm=tf.math.confusion_matrix(labels=y_true,predictions=y_pred)
    print(emotions)

    with sess.as_default():
        conf_mat=cm.eval(session=sess)
    #print(cm[0])
    fig, ax = plt.subplots()
    df_cm = pd.DataFrame(conf_mat, index = [i for i in emotions],
                  columns = [i for i in emotions])
    plt.figure(figsize = (classes,classes))
    plt.figure(2)
    print(df_cm)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    sns.heatmap(df_cm, annot=True,fmt='d',cmap=plt.cm.Blues)
    ax.set_ylim(len(df_cm) - 0.5, -0.5)
    
    plt.tight_layout()
    #plt.show()
    if classes==6:
        plt.savefig(dir_path+'conf_mat_enterface.png')
    elif classes==7:
        plt.savefig(dir_path+'conf_mat_savee.png')
    elif classes==8:
        plt.savefig(dir_path+'conf_mat_ravdess.png')
    else:
        plt.savefig(dir_path+'conf_mat_all.png')
    
 
    output_filename='conf_mat_pickle.pkl'
    with open(output_filename, 'wb') as w:
        pkl.dump(df_cm, w)

    with open('conf_mat_pickle.pkl', 'rb') as f:
        # print(f)
        df_cm = pkl.load(f)
        # plt.figure(2)
        print(df_cm)
        fig, ax = plt.subplots()
        # print(df_cm)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        sns.heatmap(df_cm, annot=True, fmt='d', cmap=plt.cm.Blues)
        ax.set_ylim(len(df_cm) - 0.5, -0.5)
        plt.tight_layout()
        plt.show()