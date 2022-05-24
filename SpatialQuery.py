#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from sklearn.neighbors import KDTree
import numpy as np


def main(argv):
    try:
        input_file = argv[0]
        task = argv[1]
        test_file = argv[2]
    except IndexError:
        print("The input arguments are in correct")
        return None
    finally:
        """task one: build kdtree"""
        if task == "1":
            df_dataset = pd.read_table(str(input_file), sep=',', usecols=[0, 1, 2])
            df_test = pd.read_table(str(test_file), header=None, sep=' ', names=['longitude', 'latitude', 'k'])
            dataset = df_dataset[['longitude', 'latitude']].to_numpy()
            df_test.longitude = df_test.longitude.str.replace("(", "").replace("'", "")
            df_test.longitude = pd.to_numeric(df_test.longitude)
            df_test.latitude = df_test.latitude.str.replace(")", "").replace("'", "")
            df_test.latitude = pd.to_numeric(df_test.latitude)
            test = df_test[['longitude', 'latitude']].to_numpy(dtype=float)
            kd_tree = KDTree(dataset, leaf_size=2)
            k_1 = df_test['k'].to_numpy()
            for i in k_1:
                j = 0
                temp_test = test[j]
                distance, indices = kd_tree.query([temp_test], k=i, return_distance=True)
                indices = indices[0]
                distance = np.array(distance[0])
                for d in distance:
                    eq_index = indices[distance == d]
                    # repeat distance
                    if len(eq_index) > 1:
                        first_index = np.where(indices == eq_index[0])[0][0]
                        last_index = np.where(indices == eq_index[-1])[0][0]
                        for index in range(first_index, last_index):
                            if indices[index] > indices[index + 1]:
                                # exchange the values
                                temp = indices[index]
                                indices[index] = indices[index + 1]
                                indices[index + 1] = temp
                print('indices:', indices)
                print("distance:", distance)
                j += 1
                with open('outputs/task1_sample_results.txt', 'a', encoding='utf-8') as f:
                    for index, v in enumerate(indices):
                        # id = index + 1
                        f.write(str(v + 1))
                        f.write('\n')
        elif task == "2":
            df_dataset = pd.read_table(str(input_file), sep=',')
            df_test = pd.read_table(str(test_file), header=None, sep=' ',
                                    names=['longitude', 'latitude', 'k', 'day_start', 'time_start', 'day_end',
                                           'time_end'])
            print(df_dataset)
            print(df_test.dayend)


if __name__ == '__main__':
    main(sys.argv[1:])
