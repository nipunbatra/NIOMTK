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
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

random.seed(42)
df = pd.read_csv('/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/01_occupancy_csv/01_winter.csv')
startdate = df.values[0][0]
end_date = df.values[-1][0]
index = pd.DatetimeIndex(start = df.values[0][0], periods = len(df)*86400, freq = '1s')
out = []
for i in range(len(df)):
    out.append(df.ix[i].values[1:])
out_1d = list(chain.from_iterable(out))
df_new = pd.Series(out_1d, index = index)
df_resampled = df_new.resample("15min")
dir_path='/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/NIOMTK_datasets'
store=HDFStore(join(dir_path, 'eco.h5'))
df=store['/building1/elec/meter1']

dataframe = df['power']['active'][startdate:end_date].resample('15min')
# print (len(dataframe.values))
# print len(df_resampled.values)
lis = []
a = df_resampled.index[len(df_resampled.values)/2]
print a
df_resampled_train = df_resampled[:a]
dataframe_train = dataframe[:a]
dataframe_train = dataframe.head(len(dataframe.values)/2)
train_length = len(dataframe_train)
train_resampled_length = len(df_resampled_train)
for i in dataframe_train:
    if np.isnan(i) == False:
        lis.append([i])
    else:
        lis.append([0])
array = np.asarray(lis)
lis = []
dataframe_test = dataframe.tail(train_length)
df_resampled_test = df_resampled.tail(train_length)
for i in dataframe_test:
    if np.isnan(i) == False:
        lis.append([i])
    else:
        lis.append([0])
array_test = np.asarray(lis)

trainingSet = df_resampled_train.values[:-1]
testSet = df_resampled_test.values
import random
split = 0.5
for i in range(len(trainingSet)):
    if random.random() < split:
        testSet[i], trainingSet[i] = trainingSet[i], testSet[i]
        array[i], array_test[i] = array_test[i], array[i]
print len(array)
print len(df_resampled_train)
classifier = svm.SVC().fit(array, df_resampled_train.values[:min(train_length, train_resampled_length)])
prediction = classifier.predict(array_test)
arr1 = np.asarray([int(i) for i in prediction])
arr2 = np.asarray([int(i) for i in df_resampled_test.values])

print (metrics.classification_report(arr1, arr2))
# def get_ground_truth(input = '/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/01_occupancy_csv/01_winter.csv', dir_path='/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/NIOMTK_datasets', key = '/building1/elec/meter1'):
# 	lis= []
# 	df = pd.read_csv('/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/01_occupancy_csv/01_winter.csv')
# 	index = pd.DatetimeIndex(start = df.values[0][0], periods = len(df)*86400, freq = '1s')
# 	start_date = df.values[0][0]
# 	end_date = df.values[-1][0]
# 	out = []
# 	for i in range(len(df)):
# 	    out.append(df.ix[i].values[1:])
# 	out_1d = list(chain.from_iterable(out))
# 	df_new = pd.Series(out_1d, index = index)
# 	df_resampled = df_new.resample("15min")
# 	store=HDFStore(join(dir_path, 'eco.h5'))
# 	df=store[key]
# 	print "Start date = ", start_date
# 	print "End date = ", end_date
# 	dataframe = df['power']['active'][start_date:end_date].resample('15min')
# 	a = df_resampled.index[len(df_resampled.values)/2]
# 	df_resampled_train = df_resampled[:a]
# 	dataframe_train = dataframe[:a]
# 	dataframe_train = dataframe.head(len(dataframe.values)/2)
# 	train_length = len(dataframe_train)
# 	train_resampled_length = len(df_resampled_train)
# 	for i in dataframe_train:
# 		if np.isnan(i) == False:
# 			lis.append([i])
# 		else:
# 			lis.append([0])
# 	array = np.asarray(lis)
# 	dataframe_test = dataframe.tail(train_length)
# 	df_resampled_test = df_resampled.tail(train_length)
# 	for i in dataframe_test:
# 		if np.isnan(i) == False:
# 			lis.append([i])
# 		else:
# 			lis.append([0])
# 	array_test = np.asarray(lis)
# 	if (len(df_resampled_train.values) != len(df_resampled_test.values)):
# 		trainingSet = df_resampled_train.values[:-1]
# 		testSet = df_resampled_test.values
# 	else:
# 		trainingSet = df_resampled_train.values
# 		testSet = df_resampled_test.values
# 	split = 0.5
# 	print "Length of training set = ", len(trainingSet)
# 	for i in range(len(trainingSet) - 1):
# 		if random.random() < split:
# 			testSet[i], trainingSet[i] = trainingSet[i], testSet[i]
# 			array[i], array_test[i] = array_test[i], array[i]
# 	neigh = KNeighborsClassifier(n_neighbors=3)
# 	neigh.fit(array, trainingSet)
# 	predictions = []
# 	for i in array_test:
# 		predictions.append(int(neigh.predict([i])))
# 	accuracy(predictions, testSet)
# 	return (df_resampled, dataframe)

# def shuffle_sets(df_resampled_train, df_resampled_test, array, array_test):
# 	print df_resampled_train.head()
# 	print df_resampled_train.values
# 	if (len(df_resampled_train.values) != len(df_resampled_test.values)):
# 		trainingSet = df_resampled_train.values[:-1]
# 		testSet = df_resampled_test.values
# 	else:
# 		trainingSet = df_resampled_train.values
# 		testSet = df_resampled_test.values
# 	split = 0.5
# 	for i in range(len(trainingSet)):
# 		if random.random() < split:
# 			testSet[i], trainingSet[i] = trainingSet[i], testSet[i]
# 			array[i], array_test[i] = array_test[i], array[i]
# 	return (trainingSet, testSet, array, array_test)

# # def get_dataframe(input = '/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/01_occupancy_csv/01_winter.csv',dir_path='/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/NIOMTK_datasets', key = '/building1/elec/meter1'):
# # 	store=HDFStore(join(dir_path, 'eco.h5'))
# # 	df=store[key]
# # 	start_date = get_ground_truth(input)[1]
# # 	end_date = get_ground_truth(input)[2]
# # 	dataframe = df['power']['active'][start_date:end_date].resample('15min')
# # 	return dataframe

# def get_training_and_test(dir_path='/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/NIOMTK_datasets', key = '/building1/elec/meter1'):
# 	lis = []
# 	dataframe = get_ground_truth()[1]
# 	df_resampled = get_ground_truth()[0]
# 	a = df_resampled.index[len(df_resampled.values)/2]
# 	df_resampled_train = df_resampled[:a]
# 	dataframe_train = dataframe[:a]
# 	dataframe_train = dataframe.head(len(dataframe.values)/2)
# 	train_length = len(dataframe_train)
# 	for i in dataframe.values[:7968]:
# 		if np.isnan(i) == False:
# 			lis.append([i])
# 		else:
# 			lis.append([0])
# 	array = np.asarray(lis)
# 	dataframe_test = dataframe.tail(train_length)
# 	df_resampled_test = df_resampled.tail(train_length)
# 	for i in dataframe_test:
# 		if np.isnan(i) == False:
# 			lis.append([i])
# 		else:
# 			lis.append([0])
# 	array_test = np.asarray(lis)
# 	return array, array_test# array = np.asarray(lis)

# # lis = []


# def KNN( array, array_test, path = '/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/NIOMTK_datasets', key = '/building1/elec/meter1'):
# 	df_resampled_train, df_resampled_test = get_training_and_test(path, key)
# 	#trainingSet, testSet, array, array_test = shuffle_sets(df_resampled_train, df_resampled_test, array, array_test)
# 	if (len(df_resampled_train.values) != len(df_resampled_test.values)):
# 		trainingSet = df_resampled_train.values[:-1]
# 		testSet = df_resampled_test.values
# 	else:
# 		trainingSet = df_resampled_train.values
# 		testSet = df_resampled_test.values
# 	split = 0.5
# 	for i in range(len(trainingSet)):
# 		if random.random() < split:
# 			testSet[i], trainingSet[i] = trainingSet[i], testSet[i]
# 			array[i], array_test[i] = array_test[i], array[i]
# 	neigh = KNeighborsClassifier(n_neighbors=3)
# 	neigh.fit(array, trainingSet)
# 	predictions = []
# 	for i in array_test:
# 		predictions.append(int(neigh.predict([i])))
# 	accuracy(predictions, testSet)

# def SVM(array, array_test, path = '/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/NIOMTK_datasets', key = '/building1/elec/meter1'):
# 	df_resampled_train, df_resampled_test = get_training_and_test(path, key)
# 	classifier = svm.SVC().fit(array, df_resampled_train.values[:min(train_length, train_resampled_length)])
# 	prediction = classifier.predict(array_test)
# 	predictions = np.asarray([int(i) for i in prediction])
# 	testSet = np.asarray([int(i) for i in df_resampled_test.values])
# 	accuracy(predictions, testSet)

# def accuracy(predictions, testSet):
# 	tp = 0
# 	tn = 0
# 	fp = 0
# 	fn = 0
# 	for x in range(len(testSet)):
# 	    if int(testSet[x]) == 1 and int(predictions[x]) == 1:
# 	        tp+=1
# 	    elif (int(testSet[x]) == 0 and int(predictions[x]) == 1):
# 	        fp+=1
# 	    elif (int(testSet[x]) == 1 and int(predictions[x]) == 0):
# 	        fn+=1
# 	    else:
# 	        tn+=1
# 	print "TP = ", tp
# 	print "TN = ", tn
# 	print "FP = ", fp
# 	print "FN = ", fn
# 	print "Accuracy = ", (tp + tn + 0.0)/(tp + tn + fp + fn)
# 	print "Precision = ", (tp + 0.0)/ (tp + fp)
# 	print "Recall = ", (tp + 0.0)/(tp + fn)


# array, array_test = get_ground_truth()