import numpy as np
import matplotlib.pyplot as plt

def build_k_indices(y, k_fold, seed):
      """build k indices for k-fold."""
      num_row = y.shape[0]
      interval = int(num_row / k_fold)
      np.random.seed(seed)
      indices = np.random.permutation(num_row)
      k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
      return np.array(k_indices)

def cross_validation(y, tx, k_fold, seed, fit_function, **fit_function_kwargs):
      """return the estimated test loss and w."""
      # get k'th subgroup in test, others in train
      k_indices = build_k_indices(y, k_fold, seed)
      loss_te = 0
      for k in range(k_fold):
            te_indice = k_indices[k]
            tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
            tr_indice = tr_indice.reshape(-1)
            y_te = y[te_indice]
            y_tr = y[tr_indice]
            x_te = tx[te_indice]
            x_tr = tx[tr_indice]

            w, loss_tr = fit_function(y_tr, x_tr, **fit_function_kwargs)
            loss_te += compute_mse(y_te, x_te, w)
      
      loss_te = loss_te/k_fold

      return w, loss_te

def compute_mse(y, tx, w):
      """compute the loss by mse."""
      e = y - tx.dot(w)
      mse = e.dot(e) / (2 * len(e))
      return mse
    
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
      loss = 0
      w = initial_w
      for n_iter in range(max_iters):
            # Compute gradient and loss
            grad= compute_gradient(y, tx, w)
            loss = compute_mse(y, tx, w)
            
            # Update w by gradient
            w = w - gamma * grad
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
      return w, loss

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