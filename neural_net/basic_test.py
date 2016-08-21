import numpy as np
import basic as basic
import pandas as pd
# load x
# load y
# wrap in np arrays
#
csv_input = pd.read_csv('../logistic_regression/iris/Iris.csv')

x = np.array( csv_input.loc[:, 'SepalLengthCm':'PetalWidthCm'])
x2 = np.array(csv_input.loc[:, 'SepalLengthCm':'SepalWidthCm'])
y = np.array(csv_input['Species'].map(lambda x: int(x == 'Iris-setosa')))

y = np.vstack((y, np.zeros_like(y))).T
n = basic.Net(x,y)
regularization_term = 0
alpha = 0.01

def cost_fn(y, hyp):
  return (-y * np.log(hyp) - (1-y) * np.log(1 - hyp)).mean()

z, a = n.build_zs_and_activations()
l_d = n.build_layer_deltas(y, a, n.get_weights())
for d in l_d:
  print d.shape

# accum_deltas = map(lambda x: np.zeros_like(x.shape), n.get_weights())
# print n.build_theta_deltas(accum_deltas, n.get_weights, z, )

# print cost_fn(y, a[-1])
# print map(lambda x: None if x is None else x.shape, a)
