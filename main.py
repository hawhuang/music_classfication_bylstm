# -*- coding: utf-8 -*-
import random
import string
import os
import sys
import numpy as np

from model import createModel
from datasetTools import getDataset
from config import slicesPath
from config import batchSize
from config import filesPerGenre
from config import nbEpoch
from config import validationRatio, testRatio
from config import sliceSize,timeratio
import pandas as pd


import keras 
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.models import load_model
#from songToData import createSlicesFromAudio

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Trains or tests the CNN", nargs='+', choices=["train","test","slice","out"])
args = parser.parse_args()

print("--------------------------")
print("| ** Config ** ")
print("| Validation ratio: {}".format(validationRatio))
print("| Test ratio: {}".format(testRatio))
print("| Slices per genre: {}".format(filesPerGenre))
print("| Slice size: {}".format(sliceSize))
print("| time ratio: {}".format(timeratio))
print("--------------------------")

if "slice" in args.mode:
	#createSlicesFromAudio()
	sys.exit()

#List genres
genres = os.listdir(slicesPath)
genres = [filename for filename in genres if os.path.isdir(slicesPath+filename)]
nbClasses = len(genres)

#Create model 
model = createModel(nbClasses, sliceSize)

if "train" in args.mode:

	#Create or load new dataset
	train_X, train_y, validation_X, validation_y  = getDataset(filesPerGenre, genres, sliceSize, validationRatio, testRatio, mode="train")

	#Define run id for graphs
	run_id = "MusicGenres - "+str(batchSize)+" "+''.join(random.SystemRandom().choice(string.ascii_uppercase) for _ in range(10))

	#Train the model
	print("[+] Training the model...")

	#model.fit(train_X, train_y, validation_split=0.0,validation_data=(validation_X,validation_y), epochs=nbEpoch, batch_size=batchSize, verbose=2)
	train_X_transform=train_X.reshape([train_X.shape[0],train_X.shape[1],train_X.shape[2],1]).transpose((0, 2, 1,3))
	print(train_X_transform.shape)
	model.fit(train_X_transform, train_y, validation_split=0.1, epochs=nbEpoch, batch_size=batchSize, verbose=1)	
	print("    Model trained!")

	#Save trained model
	print("[+] Saving the weights...")
	model.save('musicDNN.tflearn')
	print("[+] Weights saved!")

if "test" in args.mode:

	#Create or load new dataset
	test_X, test_y = getDataset(filesPerGenre, genres, sliceSize, validationRatio, testRatio, mode="test")

	#Load weights
	print("[+] Loading weights...")
	model=load_model('musicDNN.tflearn')
	print("    Weights loaded!")
	test_X_transform=test_X.reshape([test_X.shape[0],test_X.shape[1],test_X.shape[2]]).transpose((0, 2, 1))	
	testAccuracy = model.evaluate(test_X_transform, test_y)[1]
	print("[+] Test accuracy: {} ".format(testAccuracy))
	test_X_transform=test_X.reshape([test_X.shape[0],test_X.shape[1],test_X.shape[2]]).transpose((0, 2, 1))	
	prediction=model.predict_classes(test_X_transform)
	print(prediction.shape)
	print(test_y.shape)

	id=0	
	test_y_label=np.zeros( (test_y.shape[0]) )
	for i in test_y :
		test_y_label[id]=i.tolist().index(1)
		id=id+1
	table=pd.crosstab(test_y_label,prediction,rownames=['label'],colnames=['prediction'])
	print(genres)
	print(table)

if "out" in args.mode:

	#Create or load new dataset
	test_X, test_y ,test_z = getDataset(filesPerGenre, genres, sliceSize, validationRatio, testRatio, mode="out")

	#Load weights
	print("[+] Loading weights...")
	model=load_model('musicDNN.tflearn.20171022') 
	print("    Weights loaded!")
	test_X_transform=test_X.reshape([test_X.shape[0],test_X.shape[1],test_X.shape[2]]).transpose((0, 2, 1))	


	get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])

#抽样

	list_array = list(range(1,test_X.shape[0]))
	sample = random.sample(list_array, 600)  
	test_X_transform2=test_X_transform[sample,:,:]
	test_y2=test_y[sample,:]
	test_z2=test_z[sample]
# output in test mode = 0
	layer_output = get_3rd_layer_output([test_X_transform2, 0])
    
	f = open('layer_output.txt', 'w')
    
	for i in range(0,len(layer_output[0])):
		for j in range(0,len(layer_output[0][i])):
			f.write( '%s\t'%(layer_output[0][i][j]) )
		f.write( '\n' )
	f.close()

	f = open('layer_output_label.txt', 'w')
	id=0
	for i in test_y2 :
		f.write( '%s\n'%(i.tolist().index(1)) )
		id=id+1 
	f.close()    

	f = open('layer_output_id.txt', 'w')
	id=0
	for i in test_z2 :
		f.write( '%s\n'%(i))
		id=id+1 
	f.close()      
    
    







