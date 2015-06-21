import numpy as np
import pandas as pd
from numpy import *
from pandas import *
import matplotlib
import os
import os.path
from os import *
from os.path import *
import matplotlib.pyplot as plt
from itertools import chain
from sklearn import svm, datasets, metrics
import random

def get_ground_truth(input = '/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/01_occupancy_csv/01_winter.csv'):
	df = pd.read_csv('/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/01_occupancy_csv/01_winter.csv')
	index = pd.DatetimeIndex(start = df.values[0][0], periods = len(df)*86400, freq = '1s')
	out = []
	for i in range(len(df)):
	    out.append(df.ix[i].values[1:])
	out_1d = list(chain.from_iterable(out))
	df_new = pd.Series(out_1d, index = index)
	df_resampled = df_new.resample("15min")
	return df_resampled

def shuffle_sets(df_resampled_train, df_resampled_test, array, array_test):
	trainingSet = df_resampled_train.values[:-1]
	testSet = df_resampled_test.values
	split = 0.5
	for i in range(len(trainingSet)):
		if random.random() < split:
			testSet[i], trainingSet[i] = trainingSet[i], testSet[i]
			array[i], array_test[i] = array_test[i], array[i]
	return trainingSet, testSet, array, array_test

def get_training_and_test(dir_path='/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/NIOMTK_datasets'):
	lis = []
	store=HDFStore(join(dir_path, 'eco.h5'))
	df=store['/building1/elec/meter1']
	dataframe = df['power']['active'][df.values[0][0]:df.values[-1][-1].resample('15min')
	df_resampled = get_ground_truth()
	a = df_resampled.index[len(df_resampled.values)/2]
	df_resampled_train = df_resampled[:a]
	dataframe_train = dataframe[:a]
	dataframe_train = dataframe.head(len(dataframe.values)/2)
	for i in dataframe.values[:7968]:
		if np.isnan(i) == False:
			lis.append([i])
		else:
			lis.append([0])
	dataframe_test = dataframe.tail(train_length)
	df_resampled_test = df_resampled.tail(train_length)

	for i in dataframe_test:
		if np.isnan(i) == False:
			lis.append([i])
		else:
			lis.append([0])
	array_test = np.asarray(lis)
	return array, array_test
# lis = []


# array = np.asarray(lis)
def KNN(df_resampled_train, df_resampled_test, array, array_test):
	trainingSet, testSet, array, array_test = shuffle_sets(df_resampled_train, df_resampled_test, array, array_test)
	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh.fit(array, trainingSet)
	predictions = []
	for i in array_test:
		predictions.append(int(neigh.predict([i])))
	accuracy(predictions, testSet)

def SVM(df_resampled_train, df_resampled_test, array, array_test):
	classifier = svm.SVC().fit(array, df_resampled_train.values[:min(train_length, train_resampled_length)])
	prediction = classifier.predict(array_test)
	predictions = np.asarray([int(i) for i in prediction])
	testSet = np.asarray([int(i) for i in df_resampled_test.values])
	accuracy(predictions, testSet)

def accuracy(predictions, testSet):
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for x in range(len(testSet)):
	    if int(testSet[x]) == 1 and int(predictions[x]) == 1:
	        tp+=1
	    elif (int(testSet[x]) == 0 and int(predictions[x]) == 1):
	        fp+=1
	    elif (int(testSet[x]) == 1 and int(predictions[x]) == 0):
	        fn+=1
	    else:
	        tn+=1
	print "TP = ", tp
	print "TN = ", tn
	print "FP = ", fp
	print "FN = ", fn
	print "Accuracy = ", (tp + tn + 0.0)/(tp + tn + fp + fn)
	print "Precision = ", (tp + 0.0)/ (tp + fp)
	print "Recall = ", (tp + 0.0)/(tp + fn)
