import numpy as np
import basic as basic
# load x
# load y
# wrap in np arrays
#
x = np.random.randn(10, 4)
y = np.random.randn(10, 3)
n = basic.Net(x,y)
z, a = n.build_zs_and_activations()
theta = n.get_weights()
# for w in n.weights:
#   print w.shape
d = n.build_layer_deltas(n.y, a, n.weights)
# for delta in d:
#   print delta.shape
#
regularization_term = 0

def cost_fn(y, hyp):
  return (-y * np.log(hyp) - (1-y) * np.log(1 - hyp)).mean()

def run_epoch():
  z, a = n.build_zs_and_activations()
  layer_deltas = n.build_layer_deltas(y, a, n.weights)
  # partial derivative terms

  theta_deltas = []
  for i in range(0, len(n.get_weights())):
    # since we don't compute layer deltas for layer 0 (input = x), the offset is already handled
    accum_delta = a[i].T.dot(layer_deltas[i])
    theta_deltas.append(accum_delta.mean())
    # not doing regularization atm

  updated_theta = map(lambda w_d: np.subtract(w_d[0], w_d[1]), zip(n.weights, theta_deltas))
  n.set_weights(updated_theta)
  return updated_theta


for i in xrange(0, 100):
  new_theta = run_epoch()
  _, a = n.build_zs_and_activations()
  # print 'iter: {}, cost: {}, theta: {}'.format(i, (y - a[-1]).mean(), new_theta)
  print n.get_weights()[0][0][0]

#
# for i = 1; i = m; -++:
#   forward prop(xi, yi) -> get activations (a)
#   backward prop(xi, yi) -> get delta (d) terms
#   compute Dl := Dl + d_super(l+1) * (a_super(l))T