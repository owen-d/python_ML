import numpy as np
import basic as basic
import pandas as pd
from sklearn.datasets import fetch_mldata


# csv_input = pd.read_csv('../logistic_regression/iris/Iris.csv')
# x = np.array( csv_input.loc[:, 'SepalLengthCm':'PetalWidthCm'])
# y = np.atleast_2d(np.array(csv_input['Species'].map(lambda x: int(x == 'Iris-setosa'))))
# x2 = np.array(csv_input.loc[:, 'SepalLengthCm':'SepalWidthCm'])

# y_cols = []
# for entry in csv_input['Species'].unique():
#   y_cols.append(csv_input['Species'].map(lambda x: int(x == entry)))
# y = np.array(y_cols).T

num_sets = 70000
mnist = fetch_mldata('MNIST original')
x = mnist.data[:num_sets]
y = map(lambda x: int(x == 0), mnist.target[:num_sets])
y = np.reshape(y, (num_sets, 1))
# y = np.atleast_2d(mnist.target).T[:num_sets]

n = basic.Net(x,y, num_layers=2, hidden_length=8)
regularization_term = 0
alpha = 0.01

n.run(alpha=alpha, report_every=5)
print n.get_weights()
