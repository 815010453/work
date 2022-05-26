#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import pandas as pd
from sklearn.neighbors import KDTree
import numpy as np
import os


class Point(object):
    def __init__(self, x, y, attr) -> None:
        self.x = x
        self.y = y
        self.attr = {}
        for key in attr:
            self.attr[key] = attr[key]

    def __eq__(self, other: 'Point') -> bool:
        is_att_eq = True
        for key in self.attr:
            is_att_eq = is_att_eq and self.attr[key] == other.attr[key]
        return self.x == other.x and self.y == other.y and is_att_eq

    def __str__(self) -> str:
        return str(self.x) + ',' + str(self.y) + ',' + ','.join(self.attr.values())

    def __repr__(self) -> str:
        return str(self.x) + ',' + str(self.y) + ',' + ','.join(self.attr.values())

    def is_queryItem(self, querys=None) -> bool:
        if querys is None:
            return True


class Rectangle(object):
    def __init__(self, xMin, yMin, xMax, yMax) -> None:
        self.xMin = xMin
        self.yMin = yMin
        self.xMax = xMax
        self.yMax = yMax

    def is_contain(self, point: Point) -> bool:
        if self.xMin <= point.x <= self.xMax and self.yMin <= point.y <= self.yMax:
            return True
        else:
            return False

    def is_disjoin(self, other: 'Rectangle') -> bool:
        return self.xMin > other.xMax or self.xMax < other.xMin or self.yMin > other.yMax or self.yMax < other.yMin

    def split_rect(self) -> 'dict[str, Rectangle]':
        x_mid = (self.xMin + self.xMax) / 2
        y_mid = (self.yMin + self.yMax) / 2
        return {
            'NW': Rectangle(self.xMin, y_mid, x_mid, self.yMax),
            'NE': Rectangle(x_mid, y_mid, self.xMax, self.yMax),
            'SW': Rectangle(self.xMin, self.yMin, x_mid, y_mid),
            'SE': Rectangle(x_mid, self.yMin, self.xMax, y_mid)
        }


class QuadTree(object):
    def __init__(self, rect, points=[], limit=1):
        self.rect: Rectangle = rect
        self.points: list = points
        self.count: int = 0
        self.limit: int = limit
        self.children = {
            'NW': None,
            'NE': None,
            'SW': None,
            'SE': None
        }

    def insert(self, point: Point) -> None:
        self.points.append(point)
        self.count += 1
        if len(self.points) <= self.limit and self.children['NW'] is None:
            return None
        if self.children['NW'] is not None:
            for point in self.points:
                self.children[self.get_sub_tree(point)].insert(point)
                self.points = []
        else:
            if len(self.points) > self.limit:
                split_rec_dict = self.rect.split_rect()
                for key in self.children.keys():
                    self.children[key] = QuadTree(split_rec_dict[key], points=[])
                for point in self.points:
                    self.children[self.get_sub_tree(point)].insert(point)
                self.points = []

    def get_sub_tree(self, point: Point) -> str:
        rec_dict = self.rect.split_rect()
        for key in rec_dict.keys():
            if rec_dict[key].is_contain(point):
                print(key)
                return key
        return ''

    def window_query(self, rect: Rectangle, querys=None) -> ' list[Point]':
        res = []
        if not self.rect.is_disjoin(rect):
            if self.children['NW'] is not None:
                for childKey in self.children:
                    res += self.children[childKey].window_query(rect, querys)
            else:
                for pointItem in self.points:
                    if pointItem.is_queryItem(querys) and rect.is_contain(pointItem):
                        res.append(pointItem)
        return res


def main(argv):
    task = 0
    input_file = ''
    test_file = ''
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
        if task == "1":
            """task one: build kd(2-d)-tree  for query"""
            df_dataset = pd.read_table(str(input_file), sep=',', usecols=[0, 1, 2])
            df_test = pd.read_table(str(test_file), header=None, sep=' ', names=['longitude', 'latitude', 'k'])
            dataset = df_dataset[['longitude', 'latitude']].to_numpy(dtype=float)
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
                # change id_index to id
                for i1 in indices:
                    id1.append(df_dataset['id'][i1])
                indices = np.array(id1, dtype=int)
                distance = np.array(distance[0])
                for d in distance:
                    eq_id = indices[distance == d]
                    # repeat distance
                    if len(eq_id) > 1:
                        first_index = np.where(indices == eq_id[0])[0][0]
                        last_index = np.where(indices == eq_id[-1])[0][0]
                        for id_index in range(first_index, last_index):
                            if indices[id_index] > indices[id_index + 1]:
                                # exchange id
                                temp = indices[id_index]
                                indices[id_index] = indices[id_index + 1]
                                indices[id_index + 1] = temp
                j += 1
                with open('outputs/task1_sample_results.txt', 'a', encoding='utf-8') as f:
                    for v in indices:
                        f.write(str(v))
                        f.write('\n')
        elif task == "2":
            "task two: build the kd(2-d)-tree by case"
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
                for id_index, v_id in enumerate(df_dataset['id']):
                    if df_dataset['datatime'][id_index] < temp_test_datatime_begin or \
                            df_dataset['datatime'][id_index] > temp_test_datatime_end:
                        drop_indices.append(id_index)
                temp_dataset = df_dataset.drop(drop_indices)
                if temp_dataset.empty:
                    continue
                dataset = temp_dataset[['longitude', 'latitude']].to_numpy()
                # construct kd-Tree
                kd_tree = KDTree(dataset, leaf_size=2)
                distance, indices = kd_tree.query([temp_test_coord], k=t, return_distance=True)
                indices = indices[0]
                id1 = []
                # change id_index to id
                for i1 in indices:
                    id1.append(df_dataset['id'][i1])
                indices = np.array(id1, dtype=int)
                distance = np.array(distance[0])
                for d in distance:
                    eq_id = indices[distance == d]
                    # repeat distance
                    if len(eq_id) > 1:
                        first_index = np.where(indices == eq_id[0])[0][0]
                        last_index = np.where(indices == eq_id[-1])[0][0]
                        for id_index in range(first_index, last_index):
                            if indices[id_index] > indices[id_index + 1]:
                                # exchange id
                                temp = indices[id_index]
                                indices[id_index] = indices[id_index + 1]
                                indices[id_index + 1] = temp
                with open('outputs/task2_sample_results.txt', 'a', encoding='utf-8') as f:
                    for v in indices:
                        f.write(str(v))
                        f.write('\n')
                j += 1
        elif task == "3":
            """Task 3: build kd(2-d)-tree"""
            df_dataset = pd.read_table(str(input_file), sep=',', usecols=[0, 1, 2])
            df_test = pd.read_table(str(test_file), header=None, sep=' ',
                                    names=['longitude1', 'latitude1', 'longitude2', 'latitude2'])
            df_test.longitude1 = df_test.longitude1.str.replace("(", "", regex=True)
            df_test.longitude1 = pd.to_numeric(df_test.longitude1)
            test_coord1 = df_test[['longitude1', 'latitude1']].to_numpy(dtype=float)
            df_test.latitude2 = df_test.latitude2.str.replace(")", "", regex=True)
            df_test.latitude2 = pd.to_numeric(df_test.latitude2)
            test_coord2 = df_test[['longitude2', 'latitude2']].to_numpy(dtype=float)
            dataset = df_dataset[['longitude', 'latitude']].to_numpy(dtype=float)
            print(test_coord1)
            print(test_coord2)
            print(dataset)
            j = 0
            for temp_coord1 in test_coord1:
                temp_coord2 = test_coord2[j]

        # elif task == "4":
        # elif task == 5":

    if __name__ == '__main__':
        main(sys.argv[1:])
