import os
import numpy as np 
import time
import cv2
import tensorflow as tf
import keras
from keras.models import Model, load_model
import matplotlib
from keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout
from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import img_to_array, load_img
import pickle as pkl 
from collections import deque

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM, BatchNormalization
import dlib

import pyaudio
import wave
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import librosa as lib
import sys
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from multiprocessing import Process, Queue


import test_audio
import test_video
from keras.backend.tensorflow_backend import set_session
from keras import backend as K

# os.environ['KMP_DUPLICATE_LIB_OK']='True'


# NUM_PARALLEL_EXEC_UNITS = 8
# config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2, allow_soft_placement=True, device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS })
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)
# os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
# os.environ["KMP_BLOCKTIME"] = "30"
# os.environ["KMP_SETTINGS"] = "1"
# os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

def predict_emotion_audio(queue_stream, audio_model):
    # import keras
    # from keras import backend as K
    # sess = tf.compat.v1.Session()
    # #tf.keras.backend.set_session(sess1)
    # sess = tf.compat.v1.keras.backend.get_session()

    # K.tensorflow_backend.set_session(sess1)
    # print("debug model load")
    # with open('test_file.pkl', 'rb') as f:
    #     audio_element = pkl.load(f)
    
    # print("model loaded")
    count = 1
    audio_queue = []
    try:
        while not queue_stream.empty() :
            audio_element = queue_stream.get()
            audio_queue.append(audio_element)
            count +=1
        #print(len(audio_queue))
        audio_element = audio_queue.pop()
        x_test=np.transpose(audio_element['audio'])
        x_test=x_test.reshape([1,x_test.shape[0],x_test.shape[1]])
        #print(x_test.shape)
        y_pred=audio_model.predict(x_test)
        #print(y_pred)
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'neutral', 'calm']
        #print('Emotion from audio:',emotions[np.argmax(y_pred)])
        return y_pred
    except:
        #print("audio not ready")
        y_pred = []
        return y_pred
        

def process_frame(queue, new_model, LSTMmodel):
    batchImages = np.stack(queue)
    #print(np.shape(batchImages))
    to = time.time()
    features = new_model.predict(batchImages, use_multiprocessing = True)
    t1 = time.time()
    #print(t1-to)
    return features

def process_videofeed(queue,predicted_emotion, new_model, LSTMmodel):
    local_queue = deque()
    for i in range(16):
        try:
            frame = queue.get()
            local_queue.append(frame)
            #print(len(local_queue))
        except:
            print("not enough frames")
            pass
    if len(local_queue) == 16:
        #print("ready for pred")
        features = process_frame(local_queue, new_model, LSTMmodel)
        features = np.array(features)
        features = features.reshape((1,features.shape[0], features.shape[1]))
        pred = LSTMmodel.predict(features)
        i = np.argmax(pred)
        labels_mapping = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4: "Sad", 5: "Surprise"}
        #print("emotion from video", labels_mapping[i])
        #predicted_emotion = labels_mapping[i]
        predicted_emotion.put(labels_mapping[i])
        return pred


def process_video_onlyCNN(queue, predicted_emotion, new_model):
    #try
    frame = queue.get()
    pred = new_model.predict(np.expand_dims(frame, axis=0))
    i = np.argmax(pred)
    #print(i)
    labels_mapping = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5:"Sad", 6:"Surprise"}
    #print("emotion from video", labels_mapping[i])
    predicted_emotion.put(labels_mapping[i])
    #return pred
    # except:
    #     pass

def process_video_CNN_frames(frames):
    pred = 0
    for frame in frames:
        pred += new_model.predict(np.expand_dims(frame, axis=0))
    pred = pred/len(frames)
    i = np.argmax(pred)
    #labels_mapping = {0:"angry", 1:"disgust", 2:"fearful", 3:"happy", 4: "neutral", 5: "sad", 6: "surprised"}
    
    #predicted_emotion.put(labels_mapping[i])
    return pred


if __name__ == "__main__":

    model_path = "models/custom_vgg_model6.h5"
    #LSTMmodel_path = "models/exp16.h5"
    new_model = keras.models.load_model(model_path, compile = False)

    #new_model.layers.pop()
    #new_model = Model(inputs=new_model.inputs, outputs=new_model.layers[-1].output)
    #LSTMmodel = keras.models.load_model(LSTMmodel_path)

    #audio_model=keras.models.load_model('model_best.h5',compile=False)
    audio_model = 'models/model_best.h5'
    audio_model = keras.models.load_model(audio_model,compile=False)

    predicted_emotion = Queue()
    predicted_emotion.put("neutral")
    queue_frames = Queue()
    queue_stream = Queue()
    video_state = Queue()
    queue = Queue()

    p1 = Process(target = test_video.get_live_video, args = (queue_frames,queue, predicted_emotion,video_state))
    p2 = Process(target=test_audio.get_live_input, args = (queue_stream,))

    p1.start()
    print('Video Started')
    p2.start()
    print('Audio Started')
    time.sleep(0.2)
   
    
    #processes will run independently, this will continuosly take frames from the shared queue and process them
    audio_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Calm']
    video_emotions = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5:"Sad", 6:"Surprise"}
    while True:
        process_video_onlyCNN(queue, predicted_emotion, new_model)
        t0 = time.time()
        a=predict_emotion_audio(queue_stream,audio_model)
        #t1 = time.time()
        #print(t1 - t0, "process time")
        if len(a) >0:
            video_state.put("ready")
            time.sleep(0.2)
            recent_frames = queue_frames.get()
            v = process_video_CNN_frames(recent_frames)

            print('Emotion from audio:',audio_emotions[np.argmax(a)])
            print("Emotion from video:", video_emotions[np.argmax(v)])

        #t1 = time.time()
        #print(t1 - t0, "process time")

        # t2 = time.time()
        # v =process_video_onlyCNN(queue, predicted_emotion, new_model)
        # t3 = time.time()
        # print(t3 - t2, "video process time")

        #print('Emotion from audio=', a)
        
       
            v_emo = max(v[0])
            # try:
            #     v_emo=max(v[0])
            # except:
            #     pass
            a_emo=max(a[0])
            # #print(v_emo,a_emo)
            if v_emo>a_emo:
                final_emo=video_emotions[np.argmax(v)]
            else:
                final_emo = audio_emotions[np.argmax(a)]
            print('Final Emotion:',final_emo)
            #print('\n')

            t1 = time.time()
            #print(t1 - t0, "process time")

    p1.join()
    p2.join()
 



#python audio-video-live.py