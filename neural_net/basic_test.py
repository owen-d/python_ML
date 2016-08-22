import numpy as np
import basic as basic
import pandas as pd
from sklearn.datasets import fetch_mldata
# load x
# load y
# wrap in np arrays
#
# csv_input = pd.read_csv('../logistic_regression/iris/Iris.csv')

# x = np.array( csv_input.loc[:, 'SepalLengthCm':'PetalWidthCm'])
# x2 = np.array(csv_input.loc[:, 'SepalLengthCm':'SepalWidthCm'])
# y = np.atleast_2d(np.array(csv_input['Species'].map(lambda x: int(x == 'Iris-setosa'))))

num_sets = 10000
mnist = fetch_mldata('MNIST original')
x = mnist.data[:num_sets]
y = np.atleast_2d(mnist.target).T[:num_sets]

n = basic.Net(x,y, num_layers=2, hidden_length=4)
regularization_term = 0
alpha = 0.0001

n.run(alpha=alpha, report_every=5)
