# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:34:35 2020

@author: KAVITA
"""

from moviepy.editor import *
import sys
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.io import wavfile
import librosa as lib
#scipy.signal.spectrogram
import numpy as np
import librosa.display
import pickle as pkl

def get_audio_features(source_filepath,i,label):
    
    feature_files= './Data/all_features_encoded/'

   
    ##### read audio as vector
    
    audio_input,sr=lib.load(source_filepath)
    #print(sr)
    
    ##### Hamming Window and stft
    #audio_stft = librosa.stft(audio_input,window='hamm',n_fft=1024)
    #audio_db = librosa.amplitude_to_db(abs(audio_stft))
    audio_mfcc=librosa.feature.mfcc(abs(audio_input),n_mfcc=32)
    print(audio_mfcc.shape)
 
    output_filename = os.path.join(feature_files, str(i) + '.pkl')
    print(output_filename)
    out = {'class_label': label,
       'audio': audio_mfcc,
       'sr': sr}
    with open(output_filename, 'wb') as w:
        pkl.dump(out, w)


def get_savee_files():
    i=1292
    av_file_directory="./Data/ravdess-emotional-speech-audio"
    #aud_folder='./Data/Audio_clips/ravdess'
    #emotions = ['anger', 'disgust','fear', 'happiness', 'sadness', 'surprise','neutral']
    print('printing path ',av_file_directory)
    for (root,dirs,files) in os.walk(av_file_directory, topdown=True): 
        #print('for 1 ')
        for file in files:
            #print('for 2 ')

            file_path = os.path.join(root,file)
            #print(file)
            emo_recognizer=str(file).split("-")[2]
            #print(emo_recognizer)
            if str(file).split("-")[0]=="03":
                #print('file starts wit')
                if emo_recognizer == "05":
                    aud_file_name='a{}'.format(i)+'.wav'
                    label="0"
    
                elif emo_recognizer == "07":
                    aud_file_name='d{}'.format(i)+'.wav'
                    label="1"
    
                elif emo_recognizer == "06":
                    aud_file_name='f{}'.format(i)+'.wav'
                    label="2"
    
                
                elif emo_recognizer == "03":
                    aud_file_name='h{}'.format(i)+'.wav'
                    label="3"
    
                elif emo_recognizer == "04":
                    aud_file_name='sa{}'.format(i)+'.wav' 
                    label="4"
    
                elif emo_recognizer == "08":
                    aud_file_name='su{}'.format(i)+'.wav'
                    label="5"
                    
                elif emo_recognizer == "01":
                    aud_file_name='n{}'.format(i)+'.wav'
                    label="6"
                    
                elif emo_recognizer == "02":
                    aud_file_name='c{}'.format(i)+'.wav'
                    label="7"
                
            else:
                aud_file_name='not_valid'
                label="_"
            
            if aud_file_name!='not_valid':
                get_audio_features(file_path,i,label)
                i+=1
    
if __name__ == "__main__":
    
    get_savee_files()
    #python open_ravdess_folders.py
