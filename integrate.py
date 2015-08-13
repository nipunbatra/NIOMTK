import numpy as np
import pandas as pd
from numpy import *
from pandas import *
import matplotlib as plt
import os
import os.path
from os import *
from os.path import *
import matplotlib.pyplot as plt
from itertools import chain
from sklearn import svm, datasets, metrics
import random
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import pytz

results_dic = {}
knn_summer_results = []
svm_summer_results = []
knn_winter_results = []
svm_winter_results = []
def stats(testSet, predictions):
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
    accuracy = (tp + tn + 0.0)/(tp + tn + fp + fn)
    precision = (tp + 0.0)/ (tp + fn)
    recall = (tp + 0.0)/(tp + fp)
    print "Accuracy = ", accuracy
    print "Precision = ", precision
    print "Recall = ", recall
    return [accuracy, precision, recall]

def occupancy(dir_path='/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/NIOMTK_datasets', csv = '/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/01_occupancy_csv/01_summer.csv'):
    eastern = pytz.timezone('GMT')
    df = pd.read_csv(csv)
    start_date = df.values[0][0]
    end_date = df.values[-1][0]


    # print "End date = ", end_date
    # print "Length of dataframe = ", len(df)
    index = pd.DatetimeIndex(start = start_date, periods = len(df) * 86400, freq = '1s')
    index = index.tz_localize(pytz.utc).tz_convert(eastern)
    print index


    out = []
    for i in range(len(df)):
        out.append(df.ix[i].values[1:])
    out_1d = list(chain.from_iterable(out))


    df_new = pd.Series(out_1d, index = index)
    df_resampled = df_new.resample("15min")
    ground_truth = df_resampled



    store=HDFStore(join(dir_path, 'eco.h5'))
    df=store['/building1/elec/meter1']
    dataframe = df['power']['active'][start_date:end_date].resample('15min')

    id = dataframe.index
    dataframe.index = id.tz_convert(eastern)
    len_df_values = len(dataframe.values)
    len_ground_truth_values = len(df_resampled.values)




    dataframe_index = dataframe.index
    ground_truth_index = ground_truth.index
    index_intersection = dataframe_index.intersection(ground_truth_index)
    dataframe_list = []


    ground_truth_list = []
    for i in index_intersection:
        dataframe_list.append(dataframe[i])
        ground_truth_list.append(ground_truth[i])


    ground_truth = pd.Series(ground_truth_list, index_intersection)
    dataframe = pd.Series(dataframe_list, index_intersection)


    a = dataframe.index[len(ground_truth.values)/2]
    #print a
    dataframe_train = dataframe[:a]
    #print len(dataframe_train)
    ground_truth_train = ground_truth[:a]
    #print len(ground_truth_train)


    lis = []
    for i in dataframe_train:
        if np.isnan(i) == False:
            lis.append([i])
        else:
            lis.append([0])
    ###
    dataframe_train_array = np.asarray(lis)
    ground_truth_train_array = ground_truth_train.values
    ###


    dataframe_test = dataframe.tail(len(ground_truth_train))
    ground_truth_test = ground_truth.tail(len(ground_truth_train))


    lis = []
    for i in dataframe_test:
        if np.isnan(i) == False:
            lis.append([i])
        else:
            lis.append([0])
    ###
    dataframe_test_array = np.asarray(lis)
    ground_truth_test_array = ground_truth_test.values
    ###
    random.seed(42)
    split = 0.5
    for i in range(min(len(dataframe_train_array), len(dataframe_test_array))):
        if (random.random() < split):
            dataframe_train_array[i], dataframe_test_array[i] = dataframe_test_array[i], dataframe_train_array[i]
            ground_truth_train[i], ground_truth_test[i] = ground_truth_test[i], ground_truth_train[i]

    #print len(dataframe_train_array)
    print dataframe_test_array
    #print len(ground_truth_train)

    classifier = svm.SVC().fit(dataframe_train_array, ground_truth_train_array)
    prediction = classifier.predict(dataframe_test_array)

    arr1 = np.asarray([int(i) for i in prediction])
    arr2 = np.asarray([int(i) for i in ground_truth_test_array])
    print "-----SVM-----"
    print (metrics.classification_report(arr1, arr2))
    #temp = {"SVM": }
    results_dic[csv.split("/")[-1] + "_svm"] = stats(arr2, arr1)
    # if "summer" in csv:
    #     temp1 = {csv: temp}
    #     results_dic["summer"] = temp1
    # else:
    #     temp1 = {csv: temp}
    #     results_dic["winter"] = temp1
    #results_dic[csv]["SVM"] = 


    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(dataframe_train_array, ground_truth_train_array)

    print "-----KNN-----"
    predictions = []
    for i in dataframe_test_array:
        predictions.append(int(neigh.predict([i])))
    #temp = {"KNN": stats(arr2, arr1)}
    results_dic[csv.split("/")[-1] + "_knn"] = stats(ground_truth_test_array, predictions)
    # temp = {"KNN": }
    # if "summer" in csv:
    #     temp1 = {csv: temp}
    #     results_dic["summer"] = temp1
    # else:
    #     temp1 = {csv: temp}
    #     results_dic["winter"] = temp1
  #  results_dic[csv]["KNN"] = 


#path = "/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP"
path = "/Users/Rishi/Documents/Master_folder/Semester_7/BTP"
csv_files = [i for i in listdir(path) if '_csv' in i]
csv_list = []
for i in csv_files:
    directory = join(path, i)
    file_list = [j for j in listdir(directory) if ".csv" in j]
    for j in file_list:
        csv_list.append(join(directory, j))

print csv_list
#dir_path='/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/NIOMTK_datasets'
dir_path = "/Users/Rishi/Downloads"
for i in csv_list:
    occupancy(dir_path, i)
print results_dic
results = {"Summer": [{"home1": results_dic[01_summer_svm]}, {"home1": results_dic{01_summer_knn}}]}
print results
# D = results_dic
# plt.bar(range(len(D)), D.values(), align='center')
# plt.xticks(range(len(D)), list(D.keys()))

# plt.show()