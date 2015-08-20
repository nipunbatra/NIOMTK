from os import listdir
from os.path import join
from itertools import chain

import pandas as pd
from nilmtk.utils import get_datastore


def get_csv(path="/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/NIOMTK", output_filename="occupancy",
            format='HDF'):
    csv_files = [i for i in listdir(path) if '_csv' in i]
    csv_list = []
    for i in csv_files:
        directory = join(path, i)
        file_list = [j for j in listdir(directory) if ".csv" in j]
        for j in file_list:
            csv_list.append(join(directory, j))
    print (csv_list)
    print (len(csv_list))
    store = get_datastore(output_filename, format, mode='w')
    for i in range(len(csv_list)):
        dataframe = pd.read_csv(csv_list[i])
        out = []
        print "The current file is: ", csv_list[i]
        df_new = []
        for j in range(len(dataframe)):
            out.append(dataframe.ix[j].values[1:])
            out_1d = list(chain.from_iterable(out))
        index = pd.DatetimeIndex(start=dataframe.values[0][0], periods=len(out_1d), freq="1s")
        df = pd.DataFrame(out_1d, index)
        # key = Key(building=1, meter=(i + 1))
        key = "/building" + str(i) + "/elec/meter" + str(i + 1)
        if "summer" in csv_list[i]:
            key = join(str(key), "summer")
        else:
            key = join(str(key), "winter")
        store.put(key, df)


get_csv()
