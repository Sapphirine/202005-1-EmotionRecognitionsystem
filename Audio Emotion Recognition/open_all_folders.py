
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
import sys



def get_audio_features(source_filepath,destination_filepath,i,label):
    
    feature_files= "D:/Facial_Recognition/all_features_encoded"
    print(feature_files)
    print(i)

    video = VideoFileClip(source_filepath) 
    audio_part = video.audio
    audio_part.write_audiofile(destination_filepath)
    
    sample_rate, samples = wavfile.read(destination_filepath)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    
    ##### read audio as vector
    
    audio_input,sr=lib.load(destination_filepath)
    
    
    ##### Hamming Window and stft
    audio_stft = librosa.stft(audio_input,window='hamm',n_fft=1024)
    #audio_db = librosa.amplitude_to_db(abs(audio_stft))
    audio_mfcc=librosa.feature.mfcc(abs(audio_input),n_mfcc=32)
    #print(audio_mfcc.shape)
 
    output_filename = os.path.join(feature_files, str(i) + '.pkl')
    print(output_filename)
    out = {'class_label': label,
       'audio': audio_mfcc,
       'sr': sr}
    with open(output_filename, 'wb') as w:
        pkl.dump(out, w)


def get_einterface_files(i):
    count_unknown= 0
    av_file_directory = "eInterface/enterface database"
    aud_folder='.\Audio_clips\enterface'
    emotions = ['anger', 'disgust','fear', 'happiness', 'sadness', 'surprise']
    for (root,dirs,files) in os.walk("eInterface/enterface database", topdown=True): 
        for file in files:
            if str(file) != "Thumbs.db": #I put this condition because on my Mac I had a bunch of thumbnail files which we want to skip
                if not str(file).startswith('.'):
                    source_path = os.path.join(root,file)
                    print(source_path)
                    if str(source_path).find("fear") > 0:
                        aud_file_name='f{}'.format(i)+'.wav'
                        label="2"

                    elif str(source_path).find("happiness") > 0 :
                        aud_file_name='h{}'.format(i)+'.wav'
                        label="3"


                    elif str(source_path).find("sadness") > 0:
                        #print(str(root).find("sadness"))
                        aud_file_name='sa{}'.format(i)+'.wav' 
                        label="4"

                    elif str(str(source_path)).find("surprise") > 0:
                        aud_file_name='su{}'.format(i)+'.wav'
                        label="5"

                    elif str(str(source_path)).find("anger") > 0 :
                        aud_file_name='a{}'.format(i)+'.wav'
                        label="0"

                    elif str(source_path).find("disgust") > 0 :
                        aud_file_name='d{}'.format(i)+'.wav'
                        label="1"

                    else : 
                        #print(str(source_path))
                        aud_file_name = "unknown"
                        count_unknown +=1

                    if aud_file_name != "unknown":
                        #print(vid_file_name)
                        #pass
                        get_audio_features(source_path,os.path.join(aud_folder,aud_file_name),i,label)
                        i+=1





def get_savee_files():
    i=0
    av_file_directory='.\AudioVisualClip\AudioVisualClip'
    aud_folder='.\Audio_clips'
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
            
            else:
                aud_file_name='not_valid'
                label="_"
            #sprint(aud_file_name)  
            if aud_file_name!='not_valid':
                get_audio_features(file_path,os.path.join(aud_folder,aud_file_name),i,label)
                i+=1
            return i
    
if __name__ == "__main__":
    
    i=get_savee_files()
    get_einterface_files(i)
    #python open_all_folders.py
