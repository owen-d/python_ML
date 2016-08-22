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

y = np.atleast_2d(np.array(csv_input['Species'].map(lambda x: int(x == 'Iris-setosa'))))
n = basic.Net(x2,y)
regularization_term = 0
alpha = 0.001

n.run(alpha=alpha)


# for i in xrange(0, 1000000):
#   z, a = n.build_zs_and_activations()
#   cost = cost_fn(y, a[-1])
#   l_d = n.build_layer_deltas(y, a, n.get_weights())
#   t_d = n.build_theta_deltas(a, l_d)
#   new_weights = map(lambda (delta, theta): theta-(alpha * delta), zip(t_d, n.get_weights()))

#   if i % 1000 is 0:
#     print cost
#   n.set_weights(new_weights)
  # print t_d[0]
  # print n.get_weights()[0].shape
  # print t_d
