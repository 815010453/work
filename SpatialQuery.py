#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import pandas as pd
from sklearn.neighbors import KDTree
import numpy as np
import os


class Point(object):
    def __init__(self, x, y, attr=None) -> None:
        self.x = x
        self.y = y
        self.attr = attr

    def __eq__(self, other: 'Point') -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def __str__(self) -> str:
        return str(self.x) + ',' + str(self.y) + ',' + str(self.attr['id'])

    def __repr__(self) -> str:
        return str(self.x) + ',' + str(self.y) + ',' + str(self.attr['id'])


class Rectangle(object):
    def __init__(self, xMin=None, yMin=None, xMax=None, yMax=None, p1: Point = None, p2: Point = None, p3: Point = None,
                 p4: Point = None) -> None:
        # standard rectangle
        self.xMin = xMin
        self.yMin = yMin
        self.xMax = xMax
        self.yMax = yMax
        # oblique rectangle
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

    def is_contain(self, point: Point) -> bool:
        if self.p1 is None:
            # standard rectangle
            if self.xMin <= point.x <= self.xMax and self.yMin <= point.y <= self.yMax:
                return True
            else:
                return False
        else:
            # oblique rectangle
            return Rectangle.GetCross(self.p1, self.p2, point) * Rectangle.GetCross(self.p3, self.p4, point) >= 0 \
                   and Rectangle.GetCross(self.p2, self.p3, point) * Rectangle.GetCross(self.p4, self.p1, point) >= 0

    def is_disjoin(self, other) -> bool:
        if isinstance(other, Rectangle):

            if other.p1 is None:
                # self and other are standard rectangles
                return self.xMin > other.xMax or self.xMax < other.xMin \
                       or self.yMin > other.yMax or self.yMax < other.yMin

            else:
                # self is a standard rectangle and other is an oblique rectangles
                return False
        elif isinstance(other, MultiRectangle):
            # self is a standard rectangle and other are multiRectangles
            for r in other.rectangles:
                if not self.is_disjoin(r):
                    return False

            return True

    def split_rect(self) -> 'dict[str, Rectangle]':
        x_mid = (self.xMin + self.xMax) / 2
        y_mid = (self.yMin + self.yMax) / 2
        return {
            'NW': Rectangle(self.xMin, y_mid, x_mid, self.yMax),
            'NE': Rectangle(x_mid, y_mid, self.xMax, self.yMax),
            'SW': Rectangle(self.xMin, self.yMin, x_mid, y_mid),
            'SE': Rectangle(x_mid, self.yMin, self.xMax, y_mid)
        }

    @staticmethod
    def GetCross(p1: Point, p2: Point, p: Point):
        return (p2.x - p1.x) * (p.y - p1.y) - (p.x - p1.x) * (p2.y - p1.y)


class MultiRectangle(object):
    def __init__(self, points: list[Point], distance: float = 1) -> None:
        self.rectangles = []
        if len(points) < 2:
            raise "input at least two points"
        temp_points1 = points[1:]
        for index in range(0, len(temp_points1)):
            point1 = temp_points1[index]
            point2 = points[index]
            if point1.x == point2.x:
                # standard rectangle
                y_min = min([point1.y, point2.y])
                y_max = max([point1.y, point2.y])
                temp_rect = Rectangle(point1.x - distance, y_min, point1.x + distance, y_max)
                self.rectangles.append(temp_rect)
            elif point1.y == point2.y:
                # standard rectangle
                x_min = min([point1.x, point2.x])
                x_max = max([point1.x, point2.x])

                temp_rect = Rectangle(x_min, point1.y - distance, x_max, point1.y + distance)
                self.rectangles.append(temp_rect)
            else:
                # oblique rectangle
                # slope
                k = (point2.y - point1.y) / (point2.x - point1.x)
                x1 = point1.x + distance * k / (1 + k * k) ** 0.5
                x2 = point1.x - distance * k / (1 + k * k) ** 0.5
                x3 = point2.x + distance * k / (1 + k * k) ** 0.5
                x4 = point2.x - distance * k / (1 + k * k) ** 0.5
                p1 = Point(x1, point1.y - 1 / k * (x1 - point1.x))
                p2 = Point(x2, point1.y - 1 / k * (x2 - point1.x))
                p3 = Point(x3, point2.y - 1 / k * (x3 - point2.x))
                p4 = Point(x4, point2.y - 1 / k * (x4 - point2.x))
                temp_rect = Rectangle(p1=p1, p2=p2, p3=p3, p4=p4)
                self.rectangles.append(temp_rect)

    def is_contain(self, point: Point) -> bool:
        for r in self.rectangles:
            if r.is_contain(point):
                return True
        return False


class QuadTree(object):
    def __init__(self, rectangle: Rectangle, points=[], limit=1):
        self.rect: Rectangle = rectangle
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
        # repeat coordinate
        for p in self.points:
            if p == point:
                self.limit += 1
        if len(self.points) <= self.limit and self.children['NW'] is None:
            return None
        if self.children['NW'] is not None:
            for p in self.points:
                self.children[self.get_sub_tree(point)].insert(p)
            self.points = []
        else:
            if len(self.points) > self.limit:
                split_rec_dict = self.rect.split_rect()
                for key in self.children.keys():
                    self.children[key] = QuadTree(split_rec_dict[key], points=[])
                for p in self.points:
                    self.children[self.get_sub_tree(point)].insert(p)
                self.points = []

    def get_sub_tree(self, point: Point) -> str:
        rec_dict = self.rect.split_rect()
        for key in rec_dict.keys():
            if rec_dict[key].is_contain(point):
                return key
        return ''

    def window_query(self, rectangle) -> 'list[Point]':
        result = []
        if not self.rect.is_disjoin(rectangle):
            if self.children['NW'] is not None:
                for childKey in self.children:
                    result += self.children[childKey].window_query(rectangle)
            else:
                for pointItem in self.points:
                    if rectangle.is_contain(pointItem):
                        result.append(pointItem)
        return result


def main(argv):
    task = 0
    input_file = ''
    test_file = ''
    try:
        input_file = str(argv[0])
        task = int(argv[1])
        test_file = str(argv[2])
    except IndexError:
        print("The input arguments are incorrect")
        return None
    finally:
        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        if task == 1:
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
        elif task == 2:
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
            df_test['datetime_begin'] = df_test.day_start + '-' + df_test.time_start
            df_test.datetime_begin = pd.to_datetime(df_test.datetime_begin)
            df_test.day_end = df_test.day_end.str.replace('"', '')
            df_test.time_end = df_test.time_end.str.replace('"', '')
            df_test.time_end = df_test.time_end.str.replace(')', '', regex=True)
            df_test['datetime_end'] = df_test.day_end + '-' + df_test.time_end
            df_test.datetime_end = pd.to_datetime(df_test.datetime_end)
            test = df_test[['longitude', 'latitude']].to_numpy(dtype=float)
            df_dataset['datetime'] = df_dataset.date + '-' + df_dataset.time + ':00'
            df_dataset.datetime = pd.to_datetime(df_dataset.datetime)
            del df_dataset['date']
            del df_dataset['time']
            k = df_test['k'].to_numpy()
            j = 0
            for t in k:
                temp_test_coord = test[j]
                temp_test_datetime_begin = df_test['datetime_begin'][j]
                temp_test_datetime_end = df_test['datetime_end'][j]
                drop_indices = []
                for id_index, v_id in enumerate(df_dataset['id']):
                    if df_dataset['datetime'][id_index] < temp_test_datetime_begin or \
                            df_dataset['datetime'][id_index] > temp_test_datetime_end:
                        drop_indices.append(id_index)
                temp_dataset = df_dataset.drop(drop_indices)
                if temp_dataset.empty:
                    continue
                dataset = temp_dataset[['longitude', 'latitude']].to_numpy()
                # construct kd-Tree
                kd_tree = KDTree(temp_dataset, leaf_size=2)
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
        elif task == 3:
            """Task 3: build quadtree for query"""
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
            s_id = df_dataset['id'].to_numpy(dtype=int)
            x_max, y_max = np.max(dataset, axis=0)
            x_min, y_min = np.min(dataset, axis=0)
            rect_all = Rectangle(x_min, y_min, x_max, y_max)

            # construct quadtree
            q_tree = QuadTree(rect_all, [])
            for index, coord in enumerate(dataset):
                attr = {'id': s_id[index]}
                temp = Point(coord[0], coord[1], attr)
                q_tree.insert(temp)

            j = 0
            for temp_coord1 in test_coord1:
                temp_coord2 = test_coord2[j]
                window_coord = np.array([temp_coord1, temp_coord2])
                x_max, y_max = np.max(window_coord, axis=0)
                x_min, y_min = np.min(window_coord, axis=0)
                rect_query = Rectangle(x_min, y_min, x_max, y_max)
                res = q_tree.window_query(rect_query)
                res_id = []
                for i in res:
                    res_id.append(i.attr['id'])
                real_res = np.array(res_id)
                real_res = np.sort(real_res)
                j += 1
                # write file
                with open('outputs/task3_sample_results.txt', 'a', encoding='utf-8') as f:
                    for v in real_res:
                        f.write(str(v))
                        f.write('\n')
        elif task == 4:
            """Task 4: build quadtree for query and construct result by case"""
            df_dataset = pd.read_table(str(input_file), sep=',')
            df_test = pd.read_table(str(test_file), header=None, sep=' +',
                                    names=['longitude1', 'latitude1', 'longitude2', 'latitude2', 'day_start',
                                           'time_start', 'day_end',
                                           'time_end'], engine='python')
            df_test.longitude1 = df_test.longitude1.str.replace("(", "", regex=True)
            df_test.longitude1 = pd.to_numeric(df_test.longitude1)
            test_coord1 = df_test[['longitude1', 'latitude1']].to_numpy(dtype=float)
            df_test.latitude2 = df_test.latitude2.str.replace(")", "", regex=True)
            df_test.latitude2 = pd.to_numeric(df_test.latitude2)
            test_coord2 = df_test[['longitude2', 'latitude2']].to_numpy(dtype=float)
            # dataset
            dataset = df_dataset[['longitude', 'latitude']].to_numpy(dtype=float)
            s_id = df_dataset['id'].to_numpy(dtype=int)
            df_dataset['datetime'] = df_dataset.date + '-' + df_dataset.time + ':00'
            df_dataset.datetime = pd.to_datetime(df_dataset.datetime)

            df_test.day_start = df_test.day_start.str.replace('(', '', regex=True)
            df_test.day_start = df_test.day_start.str.replace('"', '')
            df_test.time_start = df_test.time_start.str.replace('"', '')
            df_test['datetime_begin'] = df_test.day_start + '-' + df_test.time_start
            df_test.datetime_begin = pd.to_datetime(df_test.datetime_begin)

            df_test.day_end = df_test.day_end.str.replace('"', '')
            df_test.time_end = df_test.time_end.str.replace('"', '')
            df_test.time_end = df_test.time_end.str.replace(')', '', regex=True)
            df_test['datetime_end'] = df_test.day_end + '-' + df_test.time_end
            df_test.datetime_end = pd.to_datetime(df_test.datetime_end)
            del df_dataset['date']
            del df_dataset['time']

            x_max, y_max = np.max(dataset, axis=0)
            x_min, y_min = np.min(dataset, axis=0)
            rect_all = Rectangle(x_min, y_min, x_max, y_max)
            # construct quadtree
            q_tree = QuadTree(rect_all, [])
            for index, coord in enumerate(dataset):
                attr = {'id': s_id[index]}
                temp = Point(coord[0], coord[1], attr)
                q_tree.insert(temp)
            j = 0
            for temp_coord1 in test_coord1:
                temp_coord2 = test_coord2[j]
                temp_test_datetime_begin = df_test['datetime_begin'][j]
                temp_test_datetime_end = df_test['datetime_end'][j]
                window_coord = np.array([temp_coord1, temp_coord2])
                x_max, y_max = np.max(window_coord, axis=0)
                x_min, y_min = np.min(window_coord, axis=0)
                rect_query = Rectangle(x_min, y_min, x_max, y_max)
                res = q_tree.window_query(rect_query)
                res_id = []
                for i in res:
                    res_id.append(i.attr['id'])
                real_res = np.array(res_id)
                real_res = np.sort(real_res)
                # query by datetime
                real_datetime_res = []
                for i in real_res:
                    id_index = np.argwhere(s_id == i)[0]
                    if temp_test_datetime_begin <= df_dataset['datetime'][id_index[0]] <= temp_test_datetime_end:
                        real_datetime_res.append(i)
                j += 1
                # write file
                with open('outputs/task4_sample_results.txt', 'a', encoding='utf-8') as f:
                    for v in real_datetime_res:
                        f.write(str(v))
                        f.write('\n')
        elif task == 5:
            """Task 5: build quadtree for query and construct result by case"""
            df_dataset = pd.read_table(str(input_file), sep=',')
            dataset = df_dataset[['longitude', 'latitude']].to_numpy(dtype=float)
            s_id = df_dataset['id'].to_numpy(dtype=int)
            df_dataset['datetime'] = df_dataset.date + '-' + df_dataset.time + ':00'
            df_dataset.datetime = pd.to_datetime(df_dataset.datetime)
            del df_dataset['date']
            del df_dataset['time']
            df_test = pd.read_table(str(test_file), header=None, sep=r'"', names=['coord_group', 'date', 'distance'])
            df_test.coord_group = df_test.coord_group.str.replace('(', '', regex=True)
            df_test.coord_group = df_test.coord_group.str.replace(')', '', regex=True)
            df_test.distance = pd.to_numeric(df_test.distance)
            dis = df_test['distance'].to_numpy(dtype=float)
            df_test['datetime_begin'] = df_test.date + '-00:00:00'
            df_test.datetime_begin = pd.to_datetime(df_test.datetime_begin)
            df_test['datetime_end'] = df_test.date + '-23:59:59'
            df_test.datetime_end = pd.to_datetime(df_test.datetime_end)
            # construct quadtree
            x_max, y_max = np.max(dataset, axis=0)
            x_min, y_min = np.min(dataset, axis=0)
            rect_all = Rectangle(x_min, y_min, x_max, y_max)

            q_tree = QuadTree(rect_all, [])
            for index, coord in enumerate(dataset):
                attr = {'id': s_id[index]}
                temp = Point(coord[0], coord[1], attr)
                q_tree.insert(temp)

            j = 0
            for d in dis:
                temp_test_datetime_begin = df_test['datetime_begin'][j]
                temp_test_datetime_end = df_test['datetime_end'][j]
                d = d * 0.01
                coord_group_str = str(df_test['coord_group'][j]).strip().split(' ')
                coord_group = []
                for i in range(0, len(coord_group_str) - 1, 2):
                    p = Point(float(coord_group_str[i]), float(coord_group_str[i + 1]))
                    coord_group.append(p)
                rect_query = MultiRectangle(coord_group, d)
                res = q_tree.window_query(rect_query)
                res_id = []
                for i in res:
                    res_id.append(i.attr['id'])
                real_res = np.array(res_id)
                real_res = np.sort(real_res)
                # query by datetime
                real_datetime_res = []
                for i in real_res:
                    id_index = np.argwhere(s_id == i)[0]
                    if temp_test_datetime_begin <= df_dataset['datetime'][id_index[0]] <= temp_test_datetime_end:
                        real_datetime_res.append(i)
                j += 1
                # write file
                with open('outputs/task5_sample_results.txt', 'a', encoding='utf-8') as f:
                    for v in real_datetime_res:
                        f.write(str(v))
                        f.write('\n')


if __name__ == '__main__':
    main(sys.argv[1:])
