import pandas as pd
import numpy as np
import Source.Categorical_Decision_Tree as dt
import Source.Numerical_Decision_Tree as ndt


def test_total_info():
    df = pd.DataFrame({'A': [1, 1, 1, 1], 'left': [1, 0, 1, 1]})
    output = dt.total_info(df)
    print(output)  # 0.811 correct value


def test_attribute_selection():
    df = pd.DataFrame({'A': [0, 1, 1, 1, 0], 'B': [1, 1, 1, 1, 0], 'left': [1, 0, 1, 1, 0]})
    output = dt.attribute_selection(df)
    print('output')
    print(output)


def test_info_of_attribute():
    df = pd.DataFrame({'A': [0, 1, 1, 1, 0], 'left': [1, 0, 1, 1, 0]})
    output = dt.info_of_attribute(df, 'A')
    print(output)  # 0.95 correct value


def test_build_decision_tree():
    df = pd.DataFrame({'A': [0, 1, 2, 1, 0], 'B': [1, 1, 1, 1, 0], 'left': [1, 0, 1, 1, 0]})
    output = dt.build_decision_tree(df)
    print(output)


def test_terminate_condition():
    df = pd.DataFrame({'A': [0, 1, 1, 1, 0], 'B': [1, 1, 1, 1, 0], 'left': [1, 1, 1, 1, 1]})
    output = dt.terminate_condition(df)
    print(output)


def test_split_attribute():
    df = pd.DataFrame({'A': [0, 1, 2, 3, 4], 'B': [1, 1, 1, 1, 1], 'left': [1, 0, 0, 1, 0]})
    output = ndt.split_attribute(df)
    print(output)

def test_numerical_decision_tree():
    df = pd.DataFrame({'A': [0, 1, 2, 3, 4], 'B': [0, 1, 1, 1, 1], 'left': [0, 1, 1, 1, 1]})
    output = ndt.split_attribute(df)
    print(output)

def test_numerical_attribute_selection():
    df = pd.DataFrame({'A': [0, 1, 2, 3, 4], 'B': [0, 1, 1, 1, 1], 'left': [0, 1, 1, 1, 1]})
    output = ndt.attribute_selection(df)
    print(output)

# test_total_info()
# test_attribute_selection()
# test_info_of_attribute()
# test_build_decision_tree()
# test_terminate_condition()
# test_split_attribute()
#test_numerical_decision_tree()
test_numerical_attribute_selection()
