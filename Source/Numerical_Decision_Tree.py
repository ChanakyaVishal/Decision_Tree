import math
import pandas as pd
import pickle as pk
from Source.Predict import accuracy_for_cross_validation
from Source.Predict import f1_for_cross_validation

data = pd.read_csv('C:\\Users\\Chanakya\\Desktop\\Sem VI\\SMAI\\Decision_tree\\Data_Set\\train.csv')

# Train-Validation data Split
data_size = int(len(data.index))
train_data_size = int(len(data.index) * 0.8)
train_data = data.iloc[0:train_data_size, :]

validation_data = data.iloc[train_data_size: data_size - 1, :]
validation_data_size = int(len(validation_data.index))


def build_decision_tree(data_points):
    terminate_value = terminate_condition(data_points)
    if terminate_value != -1:
        return terminate_value

    # Get Optimal Attribute
    attribute, split_point = attribute_selection(data_points)
    if attribute == 'salary' or attribute == 'sales' or attribute == 'promotion_last_5years' or attribute == 'Work_accident':
        unique_values = data_points[attribute].unique()
        intermediate_tree = {}
        for node in unique_values:
            new_data_points = data_points[data_points[attribute] == node]
            new_data_points = new_data_points.drop(attribute, axis=1)
            intermediate_node_value = build_decision_tree(new_data_points)
            print(node)
            intermediate_tree[node] = intermediate_node_value
    else:
        intermediate_tree = {}
        new_data_points_a = data_points[data_points[attribute] < split_point]
        new_data_points_b = data_points[data_points[attribute] >= split_point]
        intermediate_node_value_a = build_decision_tree(new_data_points_a)
        intermediate_node_value_b = build_decision_tree(new_data_points_b)
        intermediate_tree['<:' + str(split_point)] = intermediate_node_value_a
        intermediate_tree['>=:' + str(split_point)] = intermediate_node_value_b

    return {attribute: intermediate_tree}


def terminate_condition(data_points):
    feature_list = list(data_points.columns.values)

    # Terminating conditions
    if data_points.size == 0:
        return True

    elif feature_list.__len__() == 1:
        # Majority voting
        positive_points = data_points['left'][data_points.left == 1].count()
        negative_points = data_points['left'][data_points.left == 0].count()
        return positive_points > negative_points

    elif data_points[data_points['left'] == 1]['left'].size == data_points.size / data_points.columns.size:
        return True
    elif data_points[data_points['left'] == 0]['left'].size == data_points.size / data_points.columns.size:
        return False

    else:
        return -1


def attribute_selection(data_points):
    feature_list = list(data_points.columns.values)
    minimum_value = 999999999
    minimum_attribute = ''
    feature_list_len = feature_list.__len__()
    for i in range(0, feature_list_len):
        split_point_cur = -1
        if feature_list[i] == 'salary' or feature_list[i] == 'sales' or feature_list[i] == 'promotion_last_5years' or \
                feature_list[i] == 'Work_accident':
            data_points_of_attribute = data_points[[feature_list[i], 'left']]
            info_after_attribute = info_of_categorical_attribute(data_points_of_attribute, feature_list[i])[
                feature_list[i]]
        else:
            if feature_list[i] != 'left':
                info_after_attribute, attribute, split_point_cur = split_attribute(data_points, feature_list[i])

        if minimum_value > info_after_attribute:
            minimum_value = info_after_attribute
            minimum_attribute = feature_list[i]
            split_point = split_point_cur

    return minimum_attribute, split_point


def split_attribute(data_points, feature):
    """
    The function would go through all numerical attributes and determine the best split possible.
    :param data_points: This is the set of data points of a single attribute and the output column
    :param feature: The current feature under consideration
    :return: minimum_value, minimum_attribute
    """
    minimum_value = 999999999
    minimum_attribute = ''
    split_point_final = 0
    sorted_values = data_points[feature].sort_values().reset_index(drop=True)
    for i in range(sorted_values.size - 1):
        split_point = (sorted_values[i] + sorted_values[i + 1]) / 2.0

        point_set_a = data_points[data_points[feature] < split_point]
        point_set_b = data_points[data_points[feature] >= split_point]
        info_a = gini_index(point_set_a[[feature, 'left']])
        info_b = gini_index(point_set_b[[feature, 'left']])
        total_len = (point_set_a.count() + point_set_b.count())[feature]
        info_attribute = info_a * (point_set_a.count() / total_len * 1.0) + info_b * (
                point_set_b.count() / total_len * 1.0)
        if minimum_value > info_attribute[feature]:
            minimum_value = info_attribute[feature]
            minimum_attribute = feature
            split_point_final = split_point

    return minimum_value, minimum_attribute, split_point_final


def info_of_categorical_attribute(data_points, feature):
    unique_feature_values = data_points[feature].unique()
    total_len = len(data_points.index)
    info_a = 0
    for i in unique_feature_values:
        info = gini_index(data_points[data_points[feature] == i])
        info_a += info * (data_points[data_points[feature] == i].count() / total_len)
    return info_a


def gini_index(data_points):
    positive_points = data_points['left'][data_points.left == 1].count()
    negative_points = data_points['left'][data_points.left == 0].count()
    total_points = positive_points + negative_points

    if positive_points != 0:
        positive_point_ratio = positive_points / (total_points)
    else:
        positive_point_ratio = 0

    if negative_points != 0:
        negative_point_ratio = negative_points / (total_points)
    else:
        negative_point_ratio = 0

    # Formula to calculate the value
    summation = positive_point_ratio * positive_point_ratio + negative_point_ratio * negative_point_ratio
    return 1 - summation


def total_info_gain(data_points):
    # print(data_points)
    # For a binary classification problem
    positive_points = data_points['left'][data_points.left == 1].count()
    negative_points = data_points['left'][data_points.left == 0].count()
    total_points = positive_points + negative_points
    # Formula to calculate the value
    if positive_points != 0:
        positive_point_ratio = positive_points / (total_points)
        positive_log_input = positive_point_ratio
    else:
        positive_point_ratio = 0
        positive_log_input = 1

    if negative_points != 0:
        negative_point_ratio = negative_points / (total_points)
        negative_log_input = negative_point_ratio
    else:
        negative_point_ratio = 0
        negative_log_input = 1

    # print(positive_point_ratio, negative_point_ratio, positive_log_input, negative_log_input)
    return -1.0 * (positive_point_ratio * math.log2(positive_log_input) + negative_log_input * math.log2(
        negative_log_input))


output = build_decision_tree(train_data)
print(output)
accuracy_cur = accuracy_for_cross_validation(output, validation_data)
f1 = f1_for_cross_validation(output, validation_data)
print(accuracy_cur)
print(f1)

file = open('C:\\Users\\Chanakya\\Desktop\\Sem VI\\SMAI\\Decision_Tree\\Output\\numerical_output_gini.pkl', 'wb')
pk.dump(output, file)
