import numpy as np
import math

def hypothesis(x, theta):
  return np.dot(x, theta)

def cost_fn(hyp, y):
  return np.power(hyp-y, 2).mean()

def apply_gradient_descent(theta, alpha, x, y):
  hyp = hypothesis(x, theta)
  cost = cost_fn(hyp, y)

  new_theta = theta - alpha/len(hyp) * (hyp-y).dot(x)

  return (new_theta, cost)

def run(x, y, max_iterations=10000000, alpha=0.001, converge=0.00001, report_every=10000):
  x_with_bias = np.pad(np.array(x), ((0,0),(1,0)), mode='constant', constant_values=1)
  y = np.array(y)
  theta = np.zeros(x_with_bias.shape[1])

  for x in xrange(0, max_iterations):
    theta, cost = apply_gradient_descent(theta, alpha, x_with_bias, y)

    if cost < converge:
      print 'Convergence hit! Iteration #{}, cost: {}'.format(x+1, cost)
      return (theta, cost)

    if x % report_every == 0:
      print 'iteration #{}, cost {}'.format(x+1, cost)

