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
scipy.signal.spectrogram
import numpy as np
import librosa.display
import pickle as pkl

def get_audio_features(source_filepath,destination_filepath,i,label):
    
    feature_files= './Data/savee_features'

    video = VideoFileClip(source_filepath) 
    audio_part = video.audio
    audio_part.write_audiofile(destination_filepath)
    
 
    
    ##### read audio as vector
    
    audio_input,sr=lib.load(destination_filepath)
    #print(sr)
    
    ##### Hamming Window and stft
    #audio_stft = librosa.stft(audio_input,window='hamm',n_fft=1024)
    #audio_db = librosa.amplitude_to_db(abs(audio_stft))
    audio_mfcc=librosa.feature.mfcc(abs(audio_input),n_mfcc=20)
    print(audio_mfcc.shape)
 
    output_filename = os.path.join(feature_files, str(i) + '.pkl')
    print(output_filename)
    out = {'class_label': label,
       'audio': audio_mfcc,
       'sr': sr}
    with open(output_filename, 'wb') as w:
        pkl.dump(out, w)


def get_savee_files():
    i=0
    av_file_directory='./Data/AudioVisualClip/AudioVisualClip'
    aud_folder='./Data/Audio_clips/savee'
    emotions = ['anger', 'disgust','fear', 'happiness', 'sadness', 'surprise','neutral']
    for (root,dirs,files) in os.walk(av_file_directory, topdown=True): 
        for file in files:
            file_path = os.path.join(root,file)
            print(file)
            if str(file)[0] == "a":
                aud_file_name='a{}'.format(i)+'.wav'
                label="0"

            elif str(file)[0] == "d":
                aud_file_name='d{}'.format(i)+'.wav'
                label="1"

            elif str(file)[0] == "f":
                aud_file_name='f{}'.format(i)+'.wav'
                label="2"

            
            elif str(file)[0] == "h":
                aud_file_name='h{}'.format(i)+'.wav'
                label="3"

            elif str(file)[0:2] == "sa":
                aud_file_name='sa{}'.format(i)+'.wav' 
                label="4"

            elif str(file)[0:2] == "su":
                aud_file_name='su{}'.format(i)+'.wav'
                label="5"
                
            elif str(file)[0] == "n":
                aud_file_name='n{}'.format(i)+'.wav'
                label="6"
            
            else:
                aud_file_name='not_valid'
                label="_"
            #sprint(aud_file_name)  
            if aud_file_name!='not_valid':
                get_audio_features(file_path,os.path.join(aud_folder,aud_file_name),i,label)
                i+=1
    
if __name__ == "__main__":
    
    get_savee_files()
    #python open_savee_folders.py
