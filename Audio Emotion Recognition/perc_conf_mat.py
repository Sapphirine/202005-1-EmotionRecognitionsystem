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

with open('conf_mat_pickle.pkl', 'rb') as f:
    # print(f)
    df_cm = pkl.load(f)
    column=df_cm.columns
    print(column)
    #sum_c=df_cm[column].sum()

    df_cm = np.around(df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis], decimals=2)
    # plt.figure(2)
    print(df_cm)
    #print(sum_c)
    fig, ax = plt.subplots()
    # print(df_cm)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
    ax.set_ylim(len(df_cm) - 0.5, -0.5)
    plt.tight_layout()
    plt.show()