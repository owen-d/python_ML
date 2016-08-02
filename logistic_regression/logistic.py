import numpy as np
from scipy.special import expit as sigmoid

def hypothesis(x, theta):
  return sigmoid(np.dot(x, theta))

def predict_classes(input):
  return np.round(input)

def cost_fn(hyp, y):
  return (-y * np.log(hyp) - (1-y) * np.log(1 - hyp)).mean()

def apply_gradient_descent(theta, alpha, x, y):
  hyp = hypothesis(x, theta)
  cost = cost_fn(hyp, y)

  # These are the same
  # print (hyp - y).dot(x)
  # print np.sum((hyp - y) * x.T, 1)
  # print (hyp-y).shape, x.shape, (hyp-y).dot(x).shape

  updated_theta = theta - alpha / len(hyp) * (hyp - y).dot(x)

  return (updated_theta, cost)

def run(x, y, max_iterations=10000000, alpha=0.01, converge=0.00001, report_every=10000):
  x_with_bias = np.pad(np.array(x), ((0,0),(1,0)), mode='constant', constant_values=1)
  y = np.array(y)
  theta = np.zeros(x_with_bias.shape[1])

  for x in xrange(0, max_iterations):
    theta, cost = apply_gradient_descent(theta, alpha, x_with_bias, y)

    if cost < converge:
      print 'Convergence hit! Iteration #{}, cost: {}, theta: {}'.format(x+1, cost, theta)
      return (theta, cost)

    if x % report_every == 0:
      print 'iteration #{}, cost {}, theta {}'.format(x+1, cost, theta)

