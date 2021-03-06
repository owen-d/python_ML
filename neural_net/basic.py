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
def cost_fn(y, hyp):
  return (-y * np.log(hyp) - (1-y) * np.log(1 - hyp)).mean()

def sigmoid(input, deriv=False):
  sigmoided = expit(input)
  if deriv == True:
    return np.multiply(input, 1-input)
  else:
    return sigmoided

def map_shapes(input):
  return map(lambda x: x.shape, input)

class Net:
  def __init__(self, x, y, num_layers=2, hidden_length=5, default_bias=0.01, activation=sigmoid, cost_fn=cost_fn):
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

  def build_zs_and_activations(self):
    # results take the format (z, a)
    results = [(None, self.x)]
    for idx, theta in enumerate(self.weights):
      # use previous activation, defaulting to x (original features)
      prev_a = results[-1][1]
      next_z = self.forward_pass(prev_a, theta)
      next_a = self.activation(next_z)
      results.append((next_z, next_a))
    return zip(*results)

  def build_layer_deltas(self, y, activations, weights):
    # print map(lambda x: None if x is None else x.shape, weights), map(lambda x: None if x is None else x.shape, activations)

    # since we have 1 more activation than weights, zip will conveniently trim it for us. It gets used below to compute base error
    rev_theta_as = list(reversed(zip(weights, activations)))

    base_error = activations[-1] - y
    deltas = [base_error]

    # loop through all the weight/z combinations in reverse order
    for i in range(0, self.num_layers):
      # clip bias, as it is not affecting preceding layers
      theta_l = rev_theta_as[i][0][1:,:]
      # theta_l = rev_theta_as[i][0]
      g_prime = self.activation(rev_theta_as[i][1], deriv=True)
      delta_l_plus_one = deltas[i]
      delta_l = np.multiply(delta_l_plus_one.dot(theta_l.T), g_prime)
      deltas.append(delta_l)
      # deltas are dCost/dZ^l

    return list(reversed(deltas))

  def build_theta_deltas(self, activations, layer_deltas):
    thetas = self.get_weights()
    accum_deltas = map(lambda x: np.zeros_like(x), thetas)
    results = []
    # print 'layer_deltas', map_shapes(layer_deltas)
    # print 'activations:', map_shapes(activations)
    # print 'accum_deltas:', map_shapes(accum_deltas)
    # print 'thetas:', map_shapes(thetas)
    # since there is 1 less layer_delta than activation, zipping them together trims
    # the remaining activation and thus joins them in the beneficial (a^l, d^l+1) offset groups
    for a, l_d in zip(activations, layer_deltas):
      # pad bias back in (always = 1). This is because we compute the partial derivative with respect to theta:
      # dCost/dTheta = (dCost/dZ^l = delta of layer l+1)  * dZ^l/dTheta
      # ... This equals:
      # delta^l+1 * a^l (remember to pad bias back into this calc, where bias = 1 always)
      a = np.pad(a, ((0,0), (1,0)), mode='constant', constant_values=1)
      t_d = np.dot(a.T, l_d)
      results.append(t_d)
    return results

  def run(self, alpha=0.001, report_every=1000, epochs=100000, include_weights=False):
    for i in xrange(0, epochs):
      z, a = self.build_zs_and_activations()
      m = len(self.y)
      cost = cost_fn(self.y, a[-1])
      l_d = self.build_layer_deltas(self.y, a, self.get_weights())
      t_d = self.build_theta_deltas(a, l_d)
      new_weights = map(lambda (delta, theta): theta-(alpha * delta)/m, zip(t_d, self.get_weights()))

      if i % report_every is 0:
        print cost, self.get_weights() if include_weights is True else None

      self.set_weights(new_weights)
