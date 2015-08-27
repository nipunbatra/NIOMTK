from itertools import chain
import io
import os
from io import *
import pandas as pd
from numpy import *
from pandas import *
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pytz
import sys
from StringIO import StringIO
from sklearn import svm, datasets, metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from common_functions import *
from sklearn.cross_validation import train_test_split
from sklearn.metrics import *

num = 100
path_to_csv = os.path.expanduser(
    "~/CER/CER_both/CER Electricity Revised March 2012/Survey data - CSV format/Smart meters Residential pre-trial survey data.csv")

question_list = ["Question 4704: Which of the following best describes how you cook in your home",
                 "Question 410: What best describes the people you live with? READ OUT"]

classifiers_dict = {"SVM": svm.SVC(), "KNN": KNeighborsClassifier(n_neighbors=3),
                    "DT": tree.DecisionTreeClassifier(), "RF": RandomForestClassifier()}

metric_dict = {"Precision": precision_score, "Recall": recall_score,
               "MCC": matthews_corrcoef, "Accuracy": accuracy_score}


def get_list(fp, df2, question):
    lis = []
    d = fp[question]
    # print "d ", d
    if "410" in question:  # of children
        for i in list(df2.index):
            if (d[i] == 2):
                lis.append(0)
            else:
                lis.append(1)
    if "4704" in question:
        for i in list(df2.index):
            if (d[i] == 1):
                lis.append(1)
            else:
                lis.append(0)
    return lis


def classData(path_to_txt="/Users/Rishi/Downloads/file1.txt",
              path_to_csv="/Users/Rishi/Documents/Master_folder/Semester_7/BTP/CER_both/CER Electricity Revised March 2012/Survey data - CSV format/Smart meters Residential pre-trial survey data.csv",
              question="Question 43111: How many people under 15 years of age live in your home?"):
    print "File: ", path_to_txt
    print "Question: ", question
    results = []
    fp = pd.read_csv(path_to_csv)
    df = pd.read_csv(path_to_txt, sep=" ", names=["ID", "Code", "Power"], index_col=0)
    d = fp[question]
    lis = get_list(fp, df, question)
    print len(lis)
    df2_train = df.head(len(df) / 2)
    print "df2 head: ", df.head()
    ground_truth_train = lis[:len(df) / 2]
    df2_test = df.tail(len(df) / 2)
    ground_truth_test = lis[len(df) / 2:]

    dataframe_train_array = df2_train.values
    ground_truth_train_array = np.asarray(ground_truth_train)
    dataframe_test_array = np.asarray([i for i in df2_test.values])
    ground_truth_test_array = np.asarray(ground_truth_test)
    print len(dataframe_train_array)  # [:num]
    print len(ground_truth_train_array)  # [:num]
    print "abc"
    for clf_name, clf in classifiers_dict.iteritems():
        print "Classifier: ", clf_name
        print 1 in dataframe_train_array[:num]
        print 0 in dataframe_train_array[:num]
        print 1 in ground_truth_train_array[:num]
        print 0 in ground_truth_train_array[:num]
        clf.fit((dataframe_train_array[:num]), (ground_truth_train_array[:num]))
    pred = clf.predict(dataframe_test_array)
    print "Precision: ", precision_score(ground_truth_test_array[:num], pred[:num])
    print "Recall: ", recall_score(ground_truth_test_array[:num], pred[:num])
    print "Accuracy: ", accuracy_score(ground_truth_test_array[:num], pred[:num])
    print "MCC: ", matthews_corrcoef(ground_truth_test_array[:num], pred[:num])


# for metric_name, metric_func in metric_dict.iteritems():


# print clf_name, "-", metric_name, ": ", metric_func(ground_truth_test_array[:num], pred[:num])
# results.append(metric_func(ground_truth_test_array, pred))
# print (results)
files = [os.path.expanduser("~/CER/") + "file"+ str(i) + ".txt" for i in range(1, 7)]
for f in files:
    for question in question_list:
        classData(f, path_to_csv, question)
