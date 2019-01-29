from matplotlib import pyplot
import pandas as pd
import pickle as pk
import matplotlib.lines as lines

from Source.Numerical_Decision_Tree import build_decision_tree
from Source.Predict import accuracy_for_cross_validation, f1_for_cross_validation, accuracy_with_node_limit

data = pd.read_csv('C:\\Users\\Chanakya\\Desktop\\Sem VI\\SMAI\\Decision_tree\\Data_Set\\train.csv')

# Train-Validation data Split
data_size = int(len(data.index))
train_data_size = int(len(data.index) * 0.8)
train_data = data.iloc[0:train_data_size, :]

validation_data = data.iloc[train_data_size: data_size - 1, :]
validation_data_size = int(len(validation_data.index))


def plot_q4(data_points, decision_boundary_A, decision_boundary_B, decision_boundary_C, decision_boundary_D, decision_boundary_E,  decision_boundary_F,  decision_boundary_G):
    data_points = data_points[['left', 'satisfaction_level', 'time_spend_company']]
    colors = [int((i + 1) % 2) for i in data_points['left']]
    fig, ax = pyplot.subplots()
    ax.scatter(data_points['satisfaction_level'], data_points['time_spend_company'], c=colors)

    line1 = [(0, decision_boundary_A), (1, decision_boundary_A)]
    line2 = [(decision_boundary_B, decision_boundary_G), (decision_boundary_B, decision_boundary_A)]
    line3 = [(decision_boundary_C, decision_boundary_G), (decision_boundary_C, decision_boundary_A)]
    line4 = [(decision_boundary_D, decision_boundary_A), (decision_boundary_D, decision_boundary_F)]
    line5 = [(decision_boundary_E, decision_boundary_A), (decision_boundary_E, decision_boundary_F)]
    line6 = [(0, decision_boundary_F), (1, decision_boundary_F)]
    line7 = [(0, decision_boundary_G), (1, decision_boundary_G)]


    (line1_xs, line1_ys) = zip(*line1)
    (line2_xs, line2_ys) = zip(*line2)
    (line3_xs, line3_ys) = zip(*line3)
    (line4_xs, line4_ys) = zip(*line4)
    (line5_xs, line5_ys) = zip(*line5)
    (line6_xs, line6_ys) = zip(*line6)
    (line7_xs, line7_ys) = zip(*line7)



    ax.add_line(lines.Line2D(line1_xs, line1_ys, linewidth=2, color='blue'))
    ax.add_line(lines.Line2D(line2_xs, line2_ys, linewidth=2, color='red'))
    ax.add_line(lines.Line2D(line3_xs, line3_ys, linewidth=2, color='red'))
    ax.add_line(lines.Line2D(line4_xs, line4_ys, linewidth=2, color='red'))
    ax.add_line(lines.Line2D(line5_xs, line5_ys, linewidth=2, color='red'))
    ax.add_line(lines.Line2D(line6_xs, line6_ys, linewidth=2, color='blue'))
    ax.add_line(lines.Line2D(line7_xs, line7_ys, linewidth=2, color='blue'))

    pyplot.xlabel('satisfaction_level')
    pyplot.ylabel('time_spend_company')

    pyplot.show()


def plot_q5(train_data, model):
    y1 = []
    y2 = []
    x = []
    for i in range(1, 25):
        print(i)
        validation_error = accuracy_with_node_limit(model, validation_data, i)
        train_error = accuracy_with_node_limit(model, train_data, i)
        print(validation_error, train_error)
        y1.append(validation_error)
        y2.append(train_error)
        x.append(i)
    print(y1)
    print(y2)
    pyplot.plot(x, y1, color='black')
    pyplot.plot(x, y2, color='red')

    pyplot.show()

file = open("C:\\Users\\Chanakya\\Desktop\\Sem VI\\SMAI\\Decision_Tree\\Output\\numerical_output_gain.pkl", 'rb')
model = pk.load(file)

# plot_q4(train_data, 4.5, 0.46499999999999997, 0.315, 0.6950000000000001, 0.925, 6.5, 2.5)
plot_q5(train_data,model)
