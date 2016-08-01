import numpy as np
from scipy.special import expit as sigmoid

def hypothesis(x, theta):
  return sigmoid(np.dot(x, theta))

def predict_classes(input):
  return np.round(input)

def cost_fn(hyp, y):
  return -y * np.log(hyp) - (1-y) * np.log(1 - hyp)

def apply_gradient_descent(theta, alpha, x, y):
  hyp = hypothesis(x, theta)
  cost = cost_fn(hyp, y)

  # print alpha / len(hyp) * np.sum(np.transpose(x) * np.array(cost))
  print alpha / len(hyp) * np.sum(np.transpose(x) * cost)
  # print type(cost)

  updated_theta = theta - alpha / len(hyp) * np.sum(cost) * x
  return (updated_theta, np.sum(cost))

def run(x, y, iterations=100, alpha=0.1):
  x_with_bias = np.pad(np.array(x), ((0,0),(1,0)), mode='constant', constant_values=1)
  y = np.array(y)
  theta = np.zeros(x_with_bias.shape[1])

  # for x in xrange(0, iterations):
  #   theta, cost = apply_gradient_descent(theta, alpha, x_with_bias, y)
  #   print 'iteration {}, cost {}'.format(x+1, cost)

