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

def build_theta(input_len, output_len, hidden_length, hidden_num):
  weights = []
  # 1 step input->hidden
  # hidden_length-1 steps hidden->hidden
  # (for steps where destination is hidden, include a bias unit)
  # 1 step hidden->output
  #


# for i = 1; i = m; -++:
#   forward prop(xi, yi) -> get activations (a)
#   backward prop(xi, yi) -> get delta (δ) terms
#   compute Δ := Δl + δ_super(l+1) * (a_super(l))T
