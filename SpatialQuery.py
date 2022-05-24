#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import pandas as pd
from sklearn.neighbors import KDTree
import numpy as np
import os


def main(argv):
    try:
        input_file = argv[0]
        task = argv[1]
        test_file = argv[2]
    except IndexError:
        print("The input arguments are in correct")
        return None
    finally:
        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        """task one and two: build kdtree for query"""
        if task == "1":
            df_dataset = pd.read_table(str(input_file), sep=',', usecols=[0, 1, 2])
            df_test = pd.read_table(str(test_file), header=None, sep=' ', names=['longitude', 'latitude', 'k'])
            dataset = df_dataset[['longitude', 'latitude']].to_numpy()
            df_test.longitude = df_test.longitude.str.replace("(", "", regex=True).replace("'", "", regex=True)
            df_test.longitude = pd.to_numeric(df_test.longitude)
            df_test.latitude = df_test.latitude.str.replace(")", "", regex=True).replace("'", "", regex=True)
            df_test.latitude = pd.to_numeric(df_test.latitude)
            test = df_test[['longitude', 'latitude']].to_numpy(dtype=float)
            # construct kd-Tree
            kd_tree = KDTree(dataset, leaf_size=2)
            k_1 = df_test['k'].to_numpy()
            j = 0
            for i in k_1:
                temp_test = test[j]
                distance, indices = kd_tree.query([temp_test], k=i, return_distance=True)
                indices = indices[0]
                id1 = []
                # change index to id
                for i1 in indices:
                    id1.append(df_dataset['id'][i1])
                indices = np.array(id1, dtype=int)
                distance = np.array(distance[0])
                for d in distance:
                    eq_index = indices[distance == d]
                    # repeat distance
                    if len(eq_index) > 1:
                        first_index = np.where(indices == eq_index[0])[0][0]
                        last_index = np.where(indices == eq_index[-1])[0][0]
                        for index in range(first_index, last_index):
                            if indices[index] > indices[index + 1]:
                                # exchange id
                                temp = indices[index]
                                indices[index] = indices[index + 1]
                                indices[index + 1] = temp
                j += 1
                with open('outputs/task1_sample_results.txt', 'a', encoding='utf-8') as f:
                    for index, v in enumerate(indices):
                        f.write(str(v))
                        f.write('\n')
        elif task == "2":
            df_dataset = pd.read_table(str(input_file), sep=',')
            df_test = pd.read_table(str(test_file), header=None, sep=' +',
                                    names=['longitude', 'latitude', 'k', 'day_start', 'time_start', 'day_end',
                                           'time_end'], engine='python')
            df_test.longitude = df_test.longitude.str.replace("(", "", regex=True)
            df_test.longitude = df_test.longitude.str.replace("'", "")
            df_test.longitude = pd.to_numeric(df_test.longitude)
            df_test.latitude = df_test.latitude.str.replace(")", "", regex=True)
            df_test.latitude = df_test.latitude.str.replace("'", "")
            df_test.latitude = pd.to_numeric(df_test.latitude)
            df_test.day_start = df_test.day_start.str.replace('(', '', regex=True)
            df_test.day_start = df_test.day_start.str.replace('"', '')
            df_test.time_start = df_test.time_start.str.replace('"', '')
            # concat
            df_test['datatime_begin'] = df_test.day_start + '-' + df_test.time_start
            df_test.datatime_begin = pd.to_datetime(df_test.datatime_begin)
            df_test.day_end = df_test.day_end.str.replace('"', '')
            df_test.time_end = df_test.time_end.str.replace('"', '')
            df_test.time_end = df_test.time_end.str.replace(')', '', regex=True)
            df_test['datatime_end'] = df_test.day_end + '-' + df_test.time_end
            df_test.datatime_end = pd.to_datetime(df_test.datatime_end)
            test = df_test[['longitude', 'latitude']].to_numpy(dtype=float)
            df_dataset['datatime'] = df_dataset.date + '-' + df_dataset.time + ':00'
            df_dataset.datatime = pd.to_datetime(df_dataset.datatime)
            k = df_test['k'].to_numpy()
            j = 0
            for t in k:
                temp_test_coord = test[j]
                temp_test_datatime_begin = df_test['datatime_begin'][j]
                temp_test_datatime_end = df_test['datatime_end'][j]
                drop_indices = []
                for index, v_id in enumerate(df_dataset['id']):
                    if df_dataset['datatime'][index] < temp_test_datatime_begin or \
                            df_dataset['datatime'][index] > temp_test_datatime_end:
                        drop_indices.append(index)
                temp_dataset = df_dataset.drop(drop_indices)
                if temp_dataset.empty:
                    continue
                dataset = temp_dataset[['longitude', 'latitude']].to_numpy()
                # construct kd-Tree
                kd_tree = KDTree(dataset, leaf_size=2)
                distance, indices = kd_tree.query([temp_test_coord], k=t, return_distance=True)
                indices = indices[0]
                id1 = []
                # change index to id
                for i1 in indices:
                    id1.append(df_dataset['id'][i1])
                indices = np.array(id1, dtype=int)
                distance = np.array(distance[0])
                for d in distance:
                    eq_index = indices[distance == d]
                    # repeat distance
                    if len(eq_index) > 1:
                        first_index = np.where(indices == eq_index[0])[0][0]
                        last_index = np.where(indices == eq_index[-1])[0][0]
                        for index in range(first_index, last_index):
                            if indices[index] > indices[index + 1]:
                                # exchange id
                                temp = indices[index]
                                indices[index] = indices[index + 1]
                                indices[index + 1] = temp
                with open('outputs/task2_sample_results.txt', 'a', encoding='utf-8') as f:
                    for index, v in enumerate(indices):
                        f.write(str(v))
                        f.write('\n')
                j += 1
        elif task == "3":
            df_dataset = pd.read_table(str(input_file), sep=',', usecols=[0, 1, 2])
            df_test = pd.read_table(str(test_file), header=None, sep=' ',
                                    names=['longitude1', 'latitude1', 'longitude2', 'latitude2'])


if __name__ == '__main__':
    main(sys.argv[1:])
