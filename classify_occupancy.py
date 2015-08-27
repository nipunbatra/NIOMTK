from os import *
from os.path import *
from itertools import chain
import os
from sklearn import tree
import pandas as pd
from numpy import *
from pandas import *
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pytz
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from common_functions import  *
import itertools


results_dic = {}
output_path = os.path.expanduser("~/Dropbox/niomtk_data/eco/downsampled")

classifiers_dict = {"SVM": svm.SVC(), "KNN": KNeighborsClassifier(n_neighbors=3),
                    "DT": tree.DecisionTreeClassifier(),"RF":RandomForestClassifier()}

metric_dict = {"Precision":precision_score, "Recall":recall_score,
               "MCC":matthews_corrcoef, "Accuracy":accuracy_score}

def accuracy_metrics(test_gt, prediction):
    tp=None
    tn=None
    fp=None
    fn=None
    return fp

def classify_train_test_split(folder_path):
    out = {"summer":{}, "winter":{}}
    X_test_dict = {"summer":{}, "winter":{}}
    y_test_dict = {"summer":{}, "winter":{}}
    X_train_dict = {"summer":[], "winter":[]}
    y_train_dict = {"summer":[], "winter":[]}
    y_train_list = {"summer":{}, "winter":{}}
    X_train_list = {"summer":{}, "winter":{}}


    for season in ["summer", "winter"]:
        print season
        season_path = os.path.join(folder_path, season)
        print season_path
        for home in [1, 2, 3, 4, 5]:
            home_path = season_path+"/"+str(home)+".csv"
            df = pd.read_csv(home_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            df = df.between_time("06:00", "22:00")
            X = df
            y = X.pop('occupancy')
            X_train_idx,X_test_idx,y_train,y_test = train_test_split(X.index,y,test_size=0.2)
            X_train_dict[season].append(X.ix[X_train_idx])
            X_test_dict[season][home]=X.ix[X_test_idx]
            y_train_dict[season].append(y_train)
            y_test_dict[season][home]=y_test
        y_train_list[season]=list(itertools.chain(*y_train_dict[season]))
        X_train_list[season] = pd.concat(X_train_dict[season])

    for season in ["summer", "winter"]:
        for home in [1, 2, 3, 4, 5]:
            out[season][home] = {}
            for clf_name, clf in classifiers_dict.iteritems():
                out[season][home][clf_name] = {}
                clf.fit(X_train_list[season], y_train_list[season])
                pred = clf.predict(X_test_dict[season][home])
                for metric_name, metric_func in metric_dict.iteritems():
                    out[season][home][clf_name][metric_name] = metric_func(y_test_dict[season][home], pred)

    return out


def nilm_based():
    folder_path = output_path
    season = "summer"
    home=1
    season_path = os.path.join(folder_path, season)
    home_path = season_path+"/"+str(home)+".csv"
    df = pd.read_csv(home_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.between_time("06:00", "22:00")
    X = df
    y = X.pop('occupancy')



def classify_train_test_same_home(folder_path):
    out = {}
    for season in ["summer", "winter"]:
        out[season] = {}
        season_path = os.path.join(folder_path, season)
        for home in [1, 2, 3, 4, 5]:
            out[season][home] = {}
            home_path = season_path+"/"+str(home)+".csv"
            df = pd.read_csv(home_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            df = df.between_time("06:00", "22:00")
            X = df
            y = X.pop('occupancy')
            X_train_idx,X_test_idx,y_train,y_test = train_test_split(X.index,y,test_size=0.2)
            X_train = X.ix[X_train_idx]
            X_test = X.ix[X_test_idx]
            for clf_name, clf in classifiers_dict.iteritems():
                out[season][home][clf_name] = {}
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
                for metric_name, metric_func in metric_dict.iteritems():
                    out[season][home][clf_name][metric_name] = metric_func(y_test, pred)

    return out





def plot_train_single():
    out = classify_train_test_same_home(output_path)
    res = {}
    for season, season_dict in out.iteritems():
        fig, ax = plt.subplots(nrows=3)
        res[season] = {}
        for metric in metric_dict.keys():
            res[season][metric] = {}
            for home, home_results in season_dict.iteritems():
                res[season][metric][home] = pd.DataFrame(home_results).T.to_dict()[metric]
        # Plotting
    latexify(fig_height=3.2)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    summer_df = pd.DataFrame(res['summer']['Accuracy']).T
    winter_df = pd.DataFrame(res['winter']['Accuracy']).T

    summer_df.plot(ax=ax[0], rot=0, title="Summer", kind="bar").legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    winter_df.plot(ax=ax[1], rot=0, title="Winter", kind="bar", legend=False)

    format_axes(ax[0])
    format_axes(ax[1])

    ax[1].set_xlabel("Household")
    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("Accuracy")


    #plt.tight_layout()
    plt.savefig("figures/same_home_train.pdf", bbox_inches="tight")

def plot_train_all():
    out = classify_train_test_split(output_path)
    res = {}
    fig, ax = plt.subplots(nrows=3)
    for season, season_dict in out.iteritems():

        res[season] = {}
        for metric in metric_dict.keys():
            res[season][metric] = {}
            for home, home_results in season_dict.iteritems():
                res[season][metric][home] = pd.DataFrame(home_results).T.to_dict()[metric]
    # Plotting
    latexify(fig_height=3.2)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    summer_df = pd.DataFrame(res['summer']['Accuracy']).T
    winter_df = pd.DataFrame(res['winter']['Accuracy']).T

    summer_df.plot(ax=ax[0], rot=0, title="Summer", kind="bar").legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    winter_df.plot(ax=ax[1], rot=0, title="Winter", kind="bar", legend=False)

    format_axes(ax[0])
    format_axes(ax[1])

    ax[1].set_xlabel("Household")
    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("Accuracy")


    #plt.tight_layout()
    plt.savefig("figures/split.pdf", bbox_inches="tight")

def plot_train_all_drop():
    out1 = classify_train_test_same_home(output_path)
    out2 = classify_train_test_split(output_path)
    res1 = {}
    res2= {}
    for season, season_dict in out1.iteritems():

        res1[season] = {}
        for metric in metric_dict.keys():
            res1[season][metric] = {}
            for home, home_results in season_dict.iteritems():
                res1[season][metric][home] = pd.DataFrame(home_results).T.to_dict()[metric]
    for season, season_dict in out2.iteritems():

        res2[season] = {}
        for metric in metric_dict.keys():
            res2[season][metric] = {}
            for home, home_results in season_dict.iteritems():
                res2[season][metric][home] = pd.DataFrame(home_results).T.to_dict()[metric]
        # Plotting
    latexify(fig_height=3.2)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    summer_df_1 = pd.DataFrame(res1['summer']['Accuracy']).T
    winter_df_1 = pd.DataFrame(res1['winter']['Accuracy']).T

    summer_df_2 = pd.DataFrame(res2['summer']['Accuracy']).T
    winter_df_2 = pd.DataFrame(res2['winter']['Accuracy']).T

    (summer_df_1-summer_df_2).plot(ax=ax[0], rot=0, title="Summer", kind="bar").legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    (winter_df_1-winter_df_2).plot(ax=ax[1], rot=0, title="Winter", kind="bar", legend=False)

    format_axes(ax[0])
    format_axes(ax[1])

    ax[1].set_xlabel("Household")
    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("Accuracy")


    #plt.tight_layout()
    plt.savefig("figures/split.pdf", bbox_inches="tight")




