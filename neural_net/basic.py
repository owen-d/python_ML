import numpy as np
import math

#----------------- hyper params
# input units - nx1 vector
# output units - nx1 vector of classes
# HIDDEN LAYERS:
# num_layers(scalar, size(scalar representing # nodes)... would also be possible to pass array of #nodes/layer

#-----------------
# model params
# _, num_classes = y.shape
# m, num_inputs = x.shape

def build_theta(num_features, num_classes, hidden_length, num_layers):
  weights = []
  # 1 step input->hidden
  # hidden_length-1 steps hidden->hidden
  # (for steps where destination is hidden, include a bias unit)
  # 1 step hidden->output
  for x in range(0, num_layers+1):
    # first
    if x == 0:
      shape = (hidden_length, num_features)
    # last
    elif x == num_layers:
      shape = (num_classes, hidden_length+1)
    # hidden->hidden (all others)
    else :
      shape = (hidden_length, hidden_length+1)

    this_weight = np.random.randn(*shape)
    weights.append(this_weight)
  return weights

def run(x, y, num_layers, hidden_length, epochs=1):
  theta = build_theta(len(x), y.shape[1], hidden_length, num_layers)

# for i = 1; i = m; -++:
#   forward prop(xi, yi) -> get activations (a)
#   backward prop(xi, yi) -> get delta (d) terms
#   compute Dl := Dl + d_super(l+1) * (a_super(l))T
