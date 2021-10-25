import numpy as np
import matplotlib.pyplot as plt

################################################
# Loss functions
################################################

def compute_mse(y, tx, w):
    """Compute the loss of a linear model using MSE"""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


def compute_mae(e):
    """Compute the mean absolute error given the error vector"""
    return np.mean(np.abs(e))


def compute_mse_gradient(y, tx, w):
    """Compute the gradient"""
    e = y - tx.dot(w)
    grad = - tx.T.dot(e) / len(e)
    return grad


def compute_neg_log_loss(y, tx, w):
    """Compute the negative log likelihood."""
    sig = sigmoid(tx.dot(w))
    lhs = y.T.dot(np.log(sig))
    rhs = (1 - y).T.dot(np.log(1 - sig))
    return -(lhs + rhs).squeeze()


def compute_logistic_gradient(y, tx, w):
    """Compute the gradient of logisitc loss."""
    return tx.T.dot(sigmoid(tx.dot(w)) - y)


def add_l2_reg(loss_f, grad_f, lambda_):
    """Takes a loss function and a gradient function, returns the same functions but
    modified with L2 regularization added to their output"""

    def l2_loss(y, tx, w, *args, **kwargs):
        return loss_f(y, tx, w, *args, **kwargs) + lambda_ * np.linalg.norm(w)
    
    def l2_grad(y, tx, w, *args, **kwargs):
        return grad_f(y, tx, w, *args, **kwargs) + 2 * lambda_ * w
    
    return l2_loss, l2_grad


################################################
# Regression models
################################################

def gradient_descent(y, tx, initial_w, max_iters, gamma, compute_loss, compute_grad, verbose=False):
    """Performs gradient descent using initial parameters and given loss and
    gradient functions: `compute_loss` and `compute_grad` respectively"""
    
    w = initial_w
    loss = 0

    for n_iter in range(max_iters):
        grad = compute_grad(y, tx, w)
        loss = compute_loss(y, tx, w)

        w = w - gamma * grad

        if verbose:
            print(f"Gradient Descent ({n_iter}/{max_iters - 1}): loss={loss}, w={w}")
    
    return w, loss


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_loss, compute_grad,
                                batch_size=10, verbose=False):
    """Performs stochastic gradient descent using initial parameters and given 
    loss and gradient functions: `compute_loss` and `compute_grad` respectively"""
    
    w = initial_w
    loss = 0

    for n_iter, (minibatch_y, minibatch_tx) in \
        enumerate(batch_iter(y, tx, batch_size, num_batches=max_iters)):
        
        grad = compute_loss(minibatch_y, minibatch_tx, w)
        loss = compute_grad(minibatch_y, minibatch_tx, w)

        w = w - gamma * grad

        if verbose:
            print(f"Stochastic Gradient Descent ({n_iter}/{max_iters - 1}): loss={loss}, w={w}")

    return w, loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma, verbose=False):
    """Linear regression using gradient descent"""
    return gradient_descent(y, tx, initial_w, max_iters, gamma, compute_mse, 
                            compute_mse_gradient, verbose=verbose)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=10, verbose=False):
    """Linear regression using stochastic gradient descent"""
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_mse, 
                                       compute_mse_grad, batch_size=10, verbose=verbose)


def least_squares(y, tx):
    """Least squares regression using normal equations"""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    N,D = tx.shape

    aI = 2 * N * lambda_ * np.identity(D)
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent"""
    return gradient_descent(y, tx, initial_w, max_iters, gamma, 
                            compute_neg_log_loss, compute_logistic_gradient)


def logistic_regression_SGD(y, tx, initial_w, max_iters, gamma, batch_size=10):
    """Logistic regression using stochastic gradient descent"""
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_neg_log_loss, 
                                       compute_logistic_gradient, batch_size=10)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-Regularized logistic regression using gradient descent""" 
    reg_loss, reg_grad = add_l2_reg(compute_neg_log_loss, 
                                    compute_logistic_gradient,
                                    lambda_)
    
    return gradient_descent(y, tx, initial_w, max_iters, gamma, reg_loss, reg_grad)


################################################
# Cross validation
################################################

def build_k_indices(y, k_fold, seed=1):
    """Build k indices for k-fold CV
    
    Arguments:
        y -- labels of dataset
        k_fold -- number of folds 
        seed -- random number generator seed
        
    Returns: 
        an array of `k` arrays, where subarray `i` contains
        the indices of the datapoints in fold i"""
    
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, tx, k_fold, fit_function, seed=1, **fit_function_kwargs):
    """Takes a dataset and performs cross validation on it using passed training algorithm.
    
    Arguments:
    y -- labels of dataset
    tx -- features of dataset
    k_fold -- number of folds used in CV
    fit_function -- training algorithm passed as a function, with the following signature: fit_function(y, tx, **kwargs) -> (weights, training_loss)
    
    Keyword arguments:
    seed -- random number generator seed
    fit_function_kwargs -- arguments to pass to `fit_function` algorithm

    Returns:
        the average test loss over every fold
    """
    k_indices = build_k_indices(y, k_fold, seed)
    loss_te = 0

    for k in range(k_fold):
        te_indices = k_indices[k]
        tr_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)].reshape(-1)

        y_te, x_te = y[te_indices], tx[te_indices]
        y_tr, x_tr = y[tr_indices], tx[tr_indices]

        w, _ = fit_function(y_tr, x_tr, **fit_function_kwargs)
        loss_te += compute_mse(y_te, x_te, w)

    return loss_te/k_fold


################################################
# Other helpers
################################################

def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1 / (1 + np.exp(-t))


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

            
def standardize(x):  
    """Returns a standardized dataset by subtracting the mean and dividing by the std. dev for each column""" 
    return (x - x.mean(axis=0)) / x.std(axis=0)     
         
    
def display_summary_statistics(tx):
    """Takes a dataset and prints a summary of statistics to the console"""
    
    N, _ = tx.shape
    
    mean = tx.mean(axis=0)
    median = np.median(tx, axis=0)
    std = tx.std(axis=0)
    max_ = tx.max(axis=0)
    min_ = tx.min(axis=0)
    n_undef = (tx <= -999.0).sum(axis=0)
    pct_undef = n_undef / N * 100
    
    print("Column |   Mean   |  Median  | Std dev  |   Max    |    Min   | # Undefined | % Undefined ")
    for i, (m, s, med, mx, mn, nu, pu) in enumerate(zip(mean, median, std, max_, min_, n_undef, pct_undef)):
        print(f"{i:6} | {m:8.3f}   {med:8.3f}   {s:8.3f}   {mx:8.3f}   " + 
              f"{mn:8.3f}   {nu:10.3f}    {pu:10.3f}")
