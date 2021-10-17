import numpy as np
import matplotlib.pyplot as plt

def mse(e):
      '''Mean squared error'''
      return np.mean(e**2) / 2
    
def mae(e):
      '''Mean absolute error'''
      return np.mean(np.abs(e))

def compute_gradient(y, tx, w):
      '''Compute the gradient'''
      e = y - tx.dot(w)
      grad = - tx.T.dot(e) / len(e)
      return grad, e

#Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
      '''Linear regression using gradient descent'''
      # Define parameters to store w and loss
      ws = [initial_w]
      losses = []
      w = initial_w
      for n_iter in range(max_iters):
            # Compute gradient and loss
            grad, e = compute_gradient(y, tx, w)
            loss = mse(e)
            
            # Update w by gradient
            w = w - gamma * grad
            
            # Store w and loss
            ws.append(w)
            losses.append(loss)
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

      return losses, ws

def compute_stoch_gradient(y, tx, w):
      '''Compute a stochastic gradient'''
      e = y - tx.dot(w)
      grad = - tx.T.dot(e) / len(e)
      return grad, e

#Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
      "TODO"
      return "TODO"


#Least squares regression using normal equations
def least_squares(y, tx):
      """calculate the least squares."""
      a = tx.T.dot(tx)
      b = tx.T.dot(y)
      return np.linalg.solve(a, b)


#Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)


#Logistic regression using gradient descent or SGD
def logistic_regression(y, tx, initial_w, max_iters, gamma):
      "TODO"
      return "TODO"


#Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
      "TODO"
      return "TODO"