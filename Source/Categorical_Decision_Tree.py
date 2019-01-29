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

categorical_train_data = train_data[['sales', 'salary', 'Work_accident', 'promotion_last_5years', 'left']]
decision_tree = {}


def build_decision_tree(data_points, max_depth, min_samples_leaf, cur_depth):
    terminate_value = terminate_condition(data_points, max_depth, cur_depth)
    if terminate_value != -1:
        return terminate_value
    # Get Optimal Attribute
    attribute = attribute_selection(data_points)
    # print(attribute)
    unique_values = data_points[attribute].unique()
    intermediate_tree = {}
    for node in unique_values:
        new_data_points = data_points[data_points[attribute] == node]
        new_data_points = new_data_points.drop(attribute, axis=1)
        # print(new_data_points)
        intermediate_node_value = build_decision_tree(new_data_points, max_depth, min_samples_leaf, cur_depth + 1)
        if intermediate_node_value is not None:
            intermediate_tree[node] = intermediate_node_value

    return {attribute: intermediate_tree}


def terminate_condition(data_points, max_depth, cur_depth):
    feature_list = list(data_points.columns.values)

    # Terminating conditions
    if data_points.size == 0:
        # print("Sample size 0")
        return True
    elif data_points[data_points['left'] == 1]['left'].size == data_points.size / data_points.columns.size:
        return True
    elif data_points[data_points['left'] == 0]['left'].size == data_points.size / data_points.columns.size:
        return False
    elif feature_list.__len__() == 1 or cur_depth == max_depth:
        # Majority voting
        positive_points = data_points['left'][data_points.left == 1].count()
        negative_points = data_points['left'][data_points.left == 0].count()
        return positive_points > negative_points
    else:
        return -1


def attribute_selection(data_points):
    feature_list = list(data_points.columns.values)
    minimum_value = 9999999999
    minimum_attribute = ''
    feature_list_len = feature_list.__len__()
    for i in range(0, feature_list_len):
        if feature_list[i] != 'left':
            data_points_of_attribute = data_points[[feature_list[i], 'left']]
            info_after_attribute = info_of_attribute(data_points_of_attribute, feature_list[i])[feature_list[i]]
            if minimum_value > info_after_attribute:
                minimum_value = info_after_attribute
                minimum_attribute = feature_list[i]
    # print(minimum_attribute, minimum_value)
    return minimum_attribute


def info_of_attribute(data_points, feature):
    unique_feature_values = data_points[feature].unique()
    total_len = len(data_points.index)
    info_a = 0
    for i in unique_feature_values:
        temp_df = data_points[data_points[feature] == i]
        info = gini_index(temp_df)
        info_a += info * data_points[data_points[feature] == i].count() / total_len
    return info_a


def total_info_gain(data_points):
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
    return -1.0 * (positive_point_ratio * math.log2(positive_log_input) + negative_point_ratio * math.log2(
        negative_log_input))


def gini_index(data_points):
    positive_points = data_points['left'][data_points.left == 1].count()
    negative_points = data_points['left'][data_points.left == 0].count()
    total_points = positive_points + negative_points

    if positive_points != 0:
        positive_point_ratio = positive_points / total_points
    else:
        positive_point_ratio = 0

    if negative_points != 0:
        negative_point_ratio = negative_points / total_points
    else:
        negative_point_ratio = 0

    # Formula to calculate the value
    summation = positive_point_ratio * positive_point_ratio + negative_point_ratio * negative_point_ratio
    return 1 - summation


def cross_validate():
    max_accuracy = -1
    max_tree = {}
    for i in range(1, 2):
        max_depth = i + 1
        min_samples_leaf = i
        output_cur = build_decision_tree(categorical_train_data, 5, min_samples_leaf, 1)
        print(output_cur)
        accuracy_cur = accuracy_for_cross_validation(output_cur, validation_data)
        f1 = f1_for_cross_validation(output_cur, validation_data)
        print(accuracy_cur)
        if accuracy_cur > max_accuracy:
            max_accuracy = accuracy_cur
            max_tree = output_cur
    return max_tree, max_accuracy


output, accuracy_final = cross_validate()
file = open('C:\\Users\\Chanakya\\Desktop\\Sem VI\\SMAI\\Decision_Tree\\Output\\categorical_output.pkl', 'wb')

pk.dump(output, file)
print(output)
print(accuracy_final)
