
import tensorflow as tf
import re
import tensorflow_datasets as tfds
import os
import numpy as np 
import time
import cv2

import keras
from keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import LSTM, BatchNormalization
from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import img_to_array, load_img
import pickle as pkl 
from collections import deque

from keras.utils import to_categorical
from keras.models import Sequential
import dlib

import pyaudio
import wave
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import librosa as lib
import sys
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from multiprocessing import Process, Queue, Pool, Event, Pipe, Lock, Value, Array, Manager


from keras.backend.tensorflow_backend import set_session
from keras import backend as K

#import files
import test_audio
import test_video
from chatbot import *
from speechtotext import *
from texttospeech import *
from collections import deque


def preprocess_sentence(sentence):
	sentence = sentence.lower().strip()
	# creating a space between a word and the punctuation following it
	# eg: "he is a boy." => "he is a boy ."
	sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
	sentence = re.sub(r'[" "]+', " ", sentence)
	# replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
	sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
	sentence = sentence.strip()
	# adding a start and an end token to the sentence
	return sentence  
	
def evaluate(model, sentence, emotion, tokenizer, START_TOKEN, END_TOKEN):
	MAX_LENGTH = 40
	
	sentence = preprocess_sentence(sentence)

	sentence = tf.expand_dims(
	  START_TOKEN + tokenizer.encode(sentence) + END_TOKEN + tokenizer.encode(emotion), axis=0)

	output = tf.expand_dims(START_TOKEN, 0)

	for i in range(MAX_LENGTH):
		predictions = model(inputs=[sentence, output], training=False)

		# select the last word from the seq_len dimension
		predictions = predictions[:, -1:, :]
		predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

		# return the result if the predicted_id is equal to the end token
		if tf.equal(predicted_id, END_TOKEN[0]):
			break

		# concatenated the predicted_id to the output which is given to the decoder
		# as its input.
		output = tf.concat([output, predicted_id], axis=-1)

	return tf.squeeze(output, axis=0)

# def preprocess_tokenize():
# 	emotions_lst, clean_questions, clean_answers, emotions_val_lst, clean_questions_val, clean_answers_val = preprocess()
# 	tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(emotions_lst+clean_questions + clean_answers, target_vocab_size=2**13)
# 	return tokenizer

def predict(model, sentence, emotion):
	
	emotion_list = ['neutral', 'angry', 'disgust', 'fearful', 'happy', 'sad', 'surprised']
	#emotion_list, clean_questions, clean_answers = preprocess()
	# Build tokenizer using tfds for both questions and answers
	# Define start and end token to indicate the start and end of a sentence
	
	START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
	prediction = evaluate(model, sentence, emotion, tokenizer, START_TOKEN, END_TOKEN)

	predicted_sentence = tokenizer.decode(
	  [i for i in prediction if i < tokenizer.vocab_size])

	#print('Input: {}'.format(sentence))
	#print('Output: {}'.format(predicted_sentence))

	return predicted_sentence

def chatbot_predict(sentence, emotion):
	predicted_sentence = predict(model, sentence, emotion)
	return predicted_sentence

def process_video_CNN_frame(frames):
	pred = 0
	for frame in frames:
		pred += new_model.predict(np.expand_dims(frame, axis=0))
	pred = pred/len(frames)
	i = np.argmax(pred)
	labels_mapping = {0:"angry", 1:"disgust", 2:"fearful", 3:"happy", 4: "neutral", 5: "sad", 6: "surprised"}
	
	#predicted_emotion.put(labels_mapping[i])
	return pred, labels_mapping[i]

if __name__ == "__main__":
	
	model = get_model(NUM_LAYERS = 2, D_MODEL = 256, NUM_HEADS = 8, UNITS = 512, DROPOUT = 0.1)
	model_path = "audvid_models/custom_vgg_model6.h5"
	new_model = keras.models.load_model(model_path, compile = False)

	#t0 = time.time()
	#tokenizer = preprocess_tokenize()
	#t1 = time.time()
	#print("tokenizationt time", t1 - t0)

	with open("chatbot_models/model/tokenizer.pickle", "rb") as input_file:
		tokenizer = pkl.load(input_file)
   
	queue = Queue() 
	state = Queue()
	p1 = Process(target = test_video.get_live_video, args = (queue,state))
	p1.start()
	print("Welcome, enter your name")
	name = input()
	assistant_speaks("Hi "+str(name)+" How are you?") 

	while True:
		#pred, emotion = process_video_onlyCNN(queue, predicted_emotion, new_model)
		text = transcribe_audio()
		if text != None:
			print(text)
			if text == "okay bye":
				#exit out of program! 
				assistant_speaks("okay bye "+str(name)+" talk soon!") 
				break
			#send signal to process 1 to dump frames to queue
			state.put("ready")
			#wait for some time to get most recent frames
			time.sleep(0.3)
			if not queue.empty():
				#grab 5 most recent frames from queue
				recent_frames = queue.get()
				pred, emotion = process_video_CNN_frame(recent_frames)
				print(emotion)
				response = chatbot_predict(text,emotion)
				assistant_speaks(response) 
	p1.terminate()
	p1.join()
	cv2.destroyAllWindows()
	
