import numpy as np
import math
from scipy.special import expit

#----------------- hyper params
# input units - nx1 vector
# output units - nx1 vector of classes
# HIDDEN LAYERS:
# num_layers(scalar, size(scalar representing # nodes)... would also be possible to pass array of #nodes/layer

#-----------------
# model params
# _, num_classes = y.shape
# m, num_inputs = x.shape
def sigmoid(input, deriv=False):
  input = expit(input)
  if deriv == True:
    return np.multiply(input, 1-input)
  else:
    return input

def back_prop(h_x, y, weights):
  d_neg_1 = h_x - y
  d_neg_2 = np.dot(d_neg_1, theta_2) * (L2 * (1 - L2))


class Net:
  def __init__(self, x, y, num_layers=2, hidden_length=5, default_bias=0.01, activation=sigmoid):
    self.x = x
    self.y = y
    self.m = len(x)
    self.num_features = x.shape[1]
    self.num_classes = y.shape[1]
    self.activation = activation
    self.num_layers = num_layers
    self.hidden_length = hidden_length

    self.set_weights(self.build_theta(self.num_features, self.num_classes, self.hidden_length, self.num_layers))

  def get_weights(self,):
    return self.weights

  def set_weights(self, weights):
    self.weights = weights

  def build_theta(self, num_features, num_classes, hidden_length, num_layers):
    weights = []
    # 1 step input->hidden
    # hidden_length-1 steps hidden->hidden
    # (for steps where destination is hidden, include a bias unit)
    # 1 step hidden->output
    for x in range(0, num_layers+1):
      # first
      if x == 0:
        shape = (num_features+1, hidden_length)
      # last
      elif x == num_layers:
        shape = (hidden_length+1, num_classes)
      # hidden->hidden (all others)
      else :
        shape = (hidden_length+1, hidden_length)

      this_weight = np.random.randn(*shape)*0.01
      weights.append(this_weight)
    return weights

  def forward_pass(self, input, theta, bias=True):
    if bias == True:
      z = np.pad(np.array(input), ((0,0),(1,0)), mode='constant', constant_values=1).dot(theta)
    else:
      z = input.dot(theta)
    # return both the z (theta * input), as well as the activations (a, representing activation(z))
    return z

  def build_activations(self):
    zs = [self.x]
    last_idx = len(self.weights) - 1
    for idx, theta in enumerate(self.weights):
      # use previous activation, defaulting to x (original features)
      prev_a = self.activation(zs[-1])
      zs.append(self.forward_pass(prev_a, theta))
    return zs

  def back_prop(self, d_prev, theta, a_cur):
    return np.multiply(theta.dot(d_prev), np.multiply(a, (1 - a)))

  def build_deltas(self, y, zs, weights):
    # zip will trim the final z, which we don't need
    # b/c we've already calc'd the final delta as h(x)-y
    rev_theta_zs = list(reversed(zip(weights, zs)))
    # for a, b in rev_theta_zs:
    #   print a.shape, b.shape
    base_error = self.activation(zs[-1]) - y
    deltas = [base_error]

    # loop through all the weight/z combinations in reverse order
    for i in range(0, len(rev_theta_zs)):
      # clip bias, as it is not affecting preceding layers
      theta_l = rev_theta_zs[i][0][1:,:]
      g_prime = self.activation(rev_theta_zs[i][1], deriv=True)
      delta_l_plus_one = deltas[i]
      print i, delta_l_plus_one.shape, theta_l.T.shape, rev_theta_zs[i][1].shape
      delta_l = np.multiply(delta_l_plus_one.dot(theta_l.T), g_prime)
      deltas.append(delta_l)

    return list(reversed(deltas))

# for i = 1; i = m; -++:
#   forward prop(xi, yi) -> get activations (a)
#   backward prop(xi, yi) -> get delta (d) terms
#   compute Dl := Dl + d_super(l+1) * (a_super(l))T
