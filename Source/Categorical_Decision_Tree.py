import math
import pandas as pd
import numpy as np


data = pd.read_csv('C:\\Users\\Chanakya\\Desktop\\Sem VI\\SMAI\\Assignment 1\\Data_Set\\train.csv')

#Train-Validation data Split
data_size = data.size
train_data_size = int(data_size * 0.08)
train_data = data[0:train_data_size]


def build_descision_tree(data_points):
    feature_list = list(data_points.columns.values)
    feature_list_len = feature_list.__len__() - 1





def attribute_selection(data_points):
    feature_list = list(data_points.columns.values)
    minimum_value = 0
    minimum_attribute = ''
    len = feature_list.__len__()
    for i in range (0,len - 1):
        data_points_of_attribute = data_points[[feature_list[i], 'left']]
        info_after_attribute = info_of_attribute(data_points_of_attribute, feature_list[i])[feature_list[i]]
        if(minimum_value < info_after_attribute):
            minimum_value = info_after_attribute
            minimum_attribute = feature_list[i]
    return minimum_attribute


def info_of_attribute(data_points, feature):
    unique_feature_values = data_points[feature].unique()
    total_len = len(data_points.index)
    info_a = 0
    for i in unique_feature_values:
        temp_df = data_points[data_points[feature] == i]
        info = total_info(temp_df)
        info_a += info * data_points[data_points[feature] == i].count()/total_len
    return -info_a

def total_info(data_points):
    #For a binary classification problem
    positive_points = data_points['left'][data_points.left == 1].count()
    negative_points = data_points['left'][data_points.left == 0].count()
    total_points = positive_points + negative_points

    #Formula to calculate the value
    positive_point_ratio = positive_points/total_points
    negative_point_ratio = negative_points/total_points
    return -(positive_point_ratio*math.log2(positive_point_ratio + 1) + negative_point_ratio*math.log2(negative_point_ratio + 1))
