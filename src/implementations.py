import numpy as np

#Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
      "TODO"
      return "TODO"


#Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
      "TODO"
      return "TODO"


#Least squares regression using normal equations
def least_squares(y, tx):
      "TODO"
      return "TODO"


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