import numpy as np
import basic as basic
# load x
# load y
# wrap in np arrays
#
x = np.random.randn(10, 4)
y = np.random.randn(10, 3)
n = basic.Net(x,y)
a = n.build_activations()
# for w in n.weights:
#   print w.shape
d = n.build_layer_deltas(n.y, a, n.weights)
# for delta in d:
#   print delta.shape
