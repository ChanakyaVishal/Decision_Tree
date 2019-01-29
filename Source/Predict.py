import pickle as pkl
import pandas as pd


def predict(inst, tree):
    for nodes in tree.keys():
        for element in tree[nodes]:
            # print(element)
            arr = [element]
            if type(element) == str:
                arr = element.split(":")
            if arr[0] == '>=':
                value = float(inst[nodes].iloc[0]) >= float(arr[1])
                if value:
                    tree = tree[nodes][element]
            elif arr[0] == '<':
                value = float(inst[nodes].iloc[0]) < float(arr[1])
                if value:
                    tree = tree[nodes][element]
            else:
                value = inst.iloc[0][nodes]
                try:
                    tree = tree[nodes][value]
                except KeyError:
                    tree = False
                break

        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break
    return prediction


def accuracy_for_cross_validation(tree, test_data):
    correct = 0
    test_data_size = int(len(test_data.index))
    for i in range(0, test_data_size):
        predicted_class = predict(test_data.iloc[[i]], tree)
        if predicted_class == test_data.iloc[i]['left']:
            correct += 1
    accuracy = correct / test_data_size
    return accuracy


def f1_for_cross_validation(tree, test_data):
    test_data_size = int(len(test_data.index))
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, test_data_size - 1):
        predicted_class = predict(test_data.iloc[[i]], tree)
        if predicted_class == 1 and test_data.iloc[i]['left'] == 1:
            tp += 1
        elif predicted_class == 0 and test_data.iloc[i]['left'] == 0:
            tn += 1
        elif predicted_class == 1 and test_data.iloc[i]['left'] == 0:
            fp += 1
        elif predicted_class == 0 and test_data.iloc[i]['left'] == 1:
            fn += 1

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = (2 * (precision * recall)) / (precision + recall)
    print("Recall: ", recall, "Precision: ", precision, "F1_Score: ", f1_score)

    return f1_score

def predict_with_node_size(inst, tree, node_count, max_count):
    for nodes in tree.keys():
        for element in tree[nodes]:
            # print(element)
            arr = [element]
            if node_count >= max_count:
                return tree
            node_count += 1

            if type(element) == str:
                arr = element.split(":")
            if arr[0] == '>=':
                value = float(inst[nodes].iloc[0]) >= float(arr[1])
                if value:
                    tree = tree[nodes][element]
            elif arr[0] == '<':
                value = float(inst[nodes].iloc[0]) < float(arr[1])
                if value:
                    tree = tree[nodes][element]
            else:
                value = inst.iloc[0][nodes]
                try:
                    tree = tree[nodes][value]
                except KeyError:
                    tree = False
                break

        if type(tree) is dict:
            prediction = predict_with_node_size(inst, tree, node_count, max_count)
        else:
            prediction = tree
            break
    return prediction

def accuracy_with_node_limit(tree, test_data, max_node):
    correct = 0
    test_data_size = int(len(test_data.index))
    for i in range(0, test_data_size):
        predicted_class = predict_with_node_size(test_data.iloc[[i]], tree, 0, max_node)
        if predicted_class == test_data.iloc[i]['left']:
            correct += 1
    accuracy = correct / test_data_size
    return accuracy

file = open("C:\\Users\\Chanakya\\Desktop\\Sem VI\\SMAI\\Decision_Tree\\Output\\numerical_output_gini.pkl", 'rb')
model = pkl.load(file)

data = pd.read_csv('C:\\Users\\Chanakya\\Desktop\\Sem VI\\SMAI\\Decision_tree\\Data_Set\\train.csv')

data_size = int(len(data.index))
train_data_size = int(len(data.index) * 0.8)
train_data = data.iloc[0:train_data_size, :]

validation_data = data.iloc[train_data_size: data_size - 1, :]
validation_data_size = int(len(validation_data.index))

#accuracy = accuracy_for_cross_validation(model, data)
#print(accuracy)

f1 = f1_for_cross_validation(model, validation_data)
print(f1)
