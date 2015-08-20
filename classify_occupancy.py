from os import *
from os.path import *
from itertools import chain

from sklearn import tree
import pandas as pd
from numpy import *
from pandas import *
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pytz

results_dic = {}
DIR_PATH = os.path.expanduser("~/NIOMTK_datasets")

classifiers_dict = {"SVM": svm.SVC(), "KNN": KNeighborsClassifier(n_neighbors=3),
                    "DECISION TREE": tree.DecisionTreeClassifier()}


def stats(testSet, predictions):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for x in range(len(testSet)):
        if int(testSet[x]) == 1 and int(predictions[x]) == 1:
            tp += 1
        elif (int(testSet[x]) == 0 and int(predictions[x]) == 1):
            fp += 1
        elif (int(testSet[x]) == 1 and int(predictions[x]) == 0):
            fn += 1
        else:
            tn += 1
    print "TP = ", tp
    print "TN = ", tn
    print "FP = ", fp
    print "FN = ", fn
    accuracy = (tp + tn + 0.0) / (tp + tn + fp + fn)
    precision = (tp + 0.0) / (tp + fn)
    recall = (tp + 0.0) / (tp + fp)
    print "Accuracy = ", accuracy
    print "Precision = ", precision
    print "Recall = ", recall
    return [accuracy, precision, recall]


def classify_home(dir_path=DIR_PATH, csv=None):
    home_number = csv.split("/")[-1]
    results_dic[home_number] = {}
    eastern = pytz.timezone('GMT')
    df = pd.read_csv(csv)
    start_date = df.values[0][0]
    end_date = df.values[-1][0]

    index = pd.DatetimeIndex(start=start_date, periods=len(df) * 86400, freq='1s')
    index = index.tz_localize(pytz.utc).tz_convert(eastern)

    out = []
    for i in range(len(df)):
        out.append(df.ix[i].values[1:])
    out_1d = list(chain.from_iterable(out))

    df_new = pd.Series(out_1d, index=index)
    df_resampled = df_new.resample("15min")
    ground_truth = df_resampled

    store = HDFStore(join(dir_path, 'eco.h5'))
    df = store['/building1/elec/meter1']
    dataframe = df['power']['active'][start_date:end_date].resample('15min')

    id = dataframe.index
    dataframe.index = id.tz_convert(eastern)

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

    a = dataframe.index[len(ground_truth.values) / 2]
    # print a
    power_dataframe_train = dataframe[:a]
    # print len(dataframe_train)
    occupancy_ground_truth_train = ground_truth[:a]


    # Dropna?
    lis = []
    for i in power_dataframe_train:
        if np.isnan(i) == False:
            lis.append([i])
        else:
            lis.append([0])
    ###
    dataframe_train_array = np.asarray(lis)
    ground_truth_train_array = occupancy_ground_truth_train.values
    ###


    power_dataframe_test = dataframe.tail(len(occupancy_ground_truth_train))
    occupancy_ground_truth_test = ground_truth.tail(len(occupancy_ground_truth_train))

    lis = []
    for i in power_dataframe_test:
        if np.isnan(i) == False:
            lis.append([i])
        else:
            lis.append([0])
    ###
    dataframe_test_array = np.asarray(lis)
    ground_truth_test_array = occupancy_ground_truth_test.values
    ###
    random.seed(42)
    split = 0.5
    for i in range(min(len(dataframe_train_array), len(dataframe_test_array))):
        if (random.random() < split):
            dataframe_train_array[i], dataframe_test_array[i] = dataframe_test_array[i], dataframe_train_array[i]
            occupancy_ground_truth_train[i], occupancy_ground_truth_test[i] = occupancy_ground_truth_test[i], \
                                                                              occupancy_ground_truth_train[i]

    # print len(dataframe_train_array)
    print dataframe_test_array
    # print len(ground_truth_train)

    for clf_name, clf in classifiers_dict.iteritems():
        clf.fit(dataframe_train_array, ground_truth_train_array)
        prediction = clf.predict(dataframe_test_array)

        arr1 = np.asarray([int(i) for i in prediction])
        arr2 = np.asarray([int(i) for i in ground_truth_test_array])

        # temp = {"SVM": }
        results_dic[home_number][clf_name] = stats(arr2, arr1)


path = os.path.expanduser("~/csv_files")
csv_files = [i for i in listdir(path) if '_csv' in i][:2]
csv_list = []
for i in csv_files:
    directory = join(path, i)
    file_list = [j for j in listdir(directory) if ".csv" in j]
    for j in file_list:
        csv_list.append(join(directory, j))

print csv_list
for i in csv_list:
    classify_home(DIR_PATH, i)
print results_dic
