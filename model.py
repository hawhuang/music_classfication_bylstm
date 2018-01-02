# -*- coding: utf-8 -*-

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers.recurrent import LSTM, GRU 
from config import timeratio
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape ,Permute

def createModel(nbClasses,imageSize):
	print("[+] Creating model...timeratio:%d"%(timeratio))

	model=Sequential()
	#model.add(Reshape((imageSize*timeratio ,imageSize)))
	'''
	model.add(GRU(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
        batch_input_shape=(None, imageSize*timeratio ,imageSize),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
        output_dim=512,
        unroll=True,
       # return_sequences=True,
	))
	#model.add(Flatten())    
    
 
# output layer
	#model.add(Dense(units=512,activation='relu'))
	#model.add(Dropout(0.5))
	'''
	#model.add( Conv2D(input_shape=(imageSize*timeratio,128,1),filters=64,kernel_size=(3, 128) ,strides=(1, 128),activation='relu') )

	#model.add( Conv2D(filters=512,input_shape=(imageSize*timeratio,imageSize, 1) ,kernel_size=(3,3),activation='relu',padding='valid') )
	model.add( Conv2D(filters=256,input_shape=(imageSize*timeratio,imageSize, 1) ,kernel_size=(3,3),activation='relu',padding='same') ) 
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(rate=0.5))
	      
	model.add( Conv2D(filters=512,input_shape=(192,64, 256) ,kernel_size=(3,3),activation='relu',padding='same') ) 
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(rate=0.5))
	'''
    model.add(MaxPooling2D(pool_size=(2,2)))
	#model.add(Dropout(rate=0.5))

	model.add( Conv2D(filters=256,input_shape=(96,32, 1) ,kernel_size=(3,3),activation='relu',padding='same') ) 
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add( Conv2D(filters=512,input_shape=(48,16, 1) ,kernel_size=(3,3),activation='relu',padding='same') ) 
	model.add(MaxPooling2D(pool_size=(2,2)))
   ''' 
    
	model.add(Flatten())

	#model.add(Permute((1, 3,2), input_shape=(382, 1, 64)))

	l1=96
	l2=32*512
	model.add(Reshape((l1 , l2)))
	model.add(LSTM(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
        batch_input_shape=(None, l1 , l2),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
        output_dim=512,
        unroll=True,
	))
	
	
	#model.add(Dropout(0.5))
     
	model.add(Dense(units=1024,activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(units=256,activation='relu'))
	model.add(Dropout(0.5))
	
	model.add(Dense(nbClasses,activation='softmax'))

	struct=model.summary()
	
	print(struct)
    
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	
	print("    Model created!")
	

	return model
