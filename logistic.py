import numpy as np
from scipy.special import expit as sigmoid


num_features = 2
num_classifications = 1
iterations = 100

# add 1 for the bias unit
theta = np.zeros(num_features + 1)


def hypothesis(theta, x_m_array):
  return sigmoid(np.dot(x_m_array, np.transpose(theta)))

def predition_classes(input):
  return np.round(input)

def cost_fn(h, theta, x, y):
  hypothesis = h(theta, x)
  return -y * np.log(hypothesis) - (1-y) * np.log(1 - hypothesis)

def apply_gradient_descent(theta, alpha, x, y):
  m = len(x)
  hyp = np.dot(np.transpose(theta), x)
  cost = hyp - y
  updated_theta = theta - alpha / m * cost * theta
  return theta

def run(x, y):
  x_with_bias = np.pad(np.array(x), ((0,0),(1,0)), mode='constant', constant_values=1)
  theta = np.zeros(len(x_with_bias))
  hyp = hypothesis(theta, x)
