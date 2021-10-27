import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import cross
import seaborn as sns

from itertools import product

################################################
# Loss functions
################################################

def compute_mse(y, tx, w):
    """Compute the loss of a linear model using MSE"""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


def compute_rmse(y, tx, w):
    """Compute the loss of a linear model using RMSE"""
    return np.sqrt(2 * compute_mse(y, tx, w))


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
    logsig = -np.logaddexp(0, -tx.dot(w))   # == np.log(sigmoid(t))
    lognegsig = -np.logaddexp(0, tx.dot(w)) # == np.log(1 - sigmoid(t))
    
    loss = y.T.dot(logsig) + (1 - y).T.dot(lognegsig)
    return np.squeeze(-loss)


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

        w -= gamma * grad

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

        w -= gamma * grad

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
                                       compute_mse_gradient, batch_size=batch_size, verbose=verbose)


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


def logistic_regression(y, tx, initial_w, max_iters, gamma, verbose=False):
    """Logistic regression using gradient descent"""
    return gradient_descent(y, tx, initial_w, max_iters, gamma, 
                            compute_neg_log_loss, compute_logistic_gradient, verbose=verbose)


def logistic_regression_SGD(y, tx, initial_w, max_iters, gamma, batch_size=10, verbose=False):
    """Logistic regression using stochastic gradient descent"""
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_neg_log_loss, 
                                       compute_logistic_gradient, batch_size=10, verbose=verbose)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, verbose=False):
    """L2-Regularized logistic regression using gradient descent""" 
    reg_loss, reg_grad = add_l2_reg(compute_neg_log_loss, 
                                    compute_logistic_gradient,
                                    lambda_, verbose=verbose)
    
    return gradient_descent(y, tx, initial_w, max_iters, gamma, reg_loss, reg_grad)


################################################
# Cross validation and parameter search
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


def cross_validation(y, tx, k_fold, fit_function, score_function, seed=1, **fit_function_kwargs):
    """Takes a dataset and performs cross validation on it using passed training algorithm.
    
    Arguments:
    y -- labels of dataset
    tx -- features of dataset
    k_fold -- number of folds used in CV
    fit_function -- training algorithm passed as a function, with the following signature: fit_function(y, tx, **kwargs) -> (weights, training_loss)
    score_function -- function used to compute loss, averaged across folds, with the signature: loss_function(y, x, w) -> np.number
    
    Keyword arguments:
    seed -- random number generator seed
    fit_function_kwargs -- arguments to pass to `fit_function` algorithm

    Returns:
        the average test loss over every fold
    """
    k_indices = build_k_indices(y, k_fold, seed)
    score_te = 0

    for k in range(k_fold):
        te_indices = k_indices[k]
        tr_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)].reshape(-1)

        y_te, x_te = y[te_indices], tx[te_indices]
        y_tr, x_tr = y[tr_indices], tx[tr_indices]

        w, _ = fit_function(y_tr, x_tr, **fit_function_kwargs)
        score_te += score_function(y_te, x_te, w)

    return score_te/k_fold


def parameter_grid_search(y, tx, fit_function, score_function, ff_params={}, seed=1, k_fold=5,
                          verbose=False, **ff_fixed_params):
    """Performs hyperparameter search on a given model using grid search.
    
    Arguments:
    y -- labels of dataset
    tx -- features of dataset
    fit_function -- training algorithm passed as a function, with the following signature: fit_function(y, tx, **kwargs) -> (weights, training_loss)
    score_function -- function used to compute score, averaged across folds, with the signature: loss_function(y, x, w) -> np.number
    
    Keyword arguments:
    ff_params -- a dictionary of values to test. Each key is a parameter, with its respective value being an array of possible values for that parameter
    seed=1 -- random number generator seed
    k_fold=5 -- number of folds used in CV 
    verbose=False -- 
    ff_fixed_params -- arguments to pass to `fit_function` algorithm that do no change

    Returns:
        a list of dictionaries, each containing a given parameter configuration and its computed score. 
        the list is sorted from lowest to highest score
    """

    combinations = product(*ff_params.values())
    results = []

    for i, params in enumerate(combinations):
        kwargs = {param: value for param, value in zip(ff_params.keys(), params)}
        score = cross_validation(y, tx, k_fold, fit_function, score_function, seed=seed,
                                 **kwargs, **ff_fixed_params)

        results.append({"params": kwargs, "score": score})

        if verbose:
            print(f"Parameter combination {i}:")
            print(f"\tParams: {kwargs}")
            print(f"\tScore: {score}")

    return sorted(results, key=lambda x: x["score"])
    

################################################
# Other helpers
################################################

#def sigmoid(t):
#    """apply the sigmoid function on t."""
#    return 1.0 / (1.0 + np.exp(-t))

def sigmoid(t):
    "Numerically stable sigmoid function."
    return np.piecewise(t, [t > 0], 
                        [lambda i: 1 / (1 + np.exp(-i)), 
                         lambda i: np.exp(i) / (1 + np.exp(i))])



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

            
def standardize(x, mean=None, std=None):  
    """
    Takes and dataset and returns a standardized dataset, a mean, and a std. dev. 
    If no mean/std is passed to the function, the metrics are calulated on the dataset.
    Standardizing is done by subtracting the mean and dividing by the std. dev for each column, """ 
    
    mean = mean if mean is not None else x.mean(axis=0)
    std = std if std is not None else x.std(axis=0) 
    
    return (x - mean) / std, mean, std
         
    
def display_summary_statistics(tx, column_names=None):
    """Takes a dataset and prints a summary of statistics to the console"""
    
    N, D = tx.shape
    
    mean = tx.mean(axis=0)
    median = np.median(tx, axis=0)
    std = tx.std(axis=0)
    max_ = tx.max(axis=0)
    min_ = tx.min(axis=0)
    n_undef = (tx <= -999.0).sum(axis=0)
    pct_undef = (tx <= -999.0).mean(axis=0) * 100

    column_names = column_names if column_names is not None else range(D)
    
    print("   Column                      |   Mean   |  Median  | Std dev  |   Max    |    Min   | # Undefined | % Undef ")
    for i, (col, m, med, s, mx, mn, nu, pu) in enumerate(zip(column_names, mean, median, std, max_, min_, n_undef, pct_undef)):
        print(f"{i:2}-{col:27} | {m:8.3f}   {med:8.3f}   {s:8.3f}   {mx:8.3f}   " + 
              f"{mn:8.3f}   {nu:10.3f}    {pu:7.3f}")

def plot_corr_matrix(x, column_names):
    """Computes the pearson correlation matrix between features
    and plots it"""
    
    # Compute correlation matrix
    corr = np.corrcoef(x, rowvar=False)

    # Setup triangular mask and colour palette
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    f, ax = plt.subplots(figsize=(30, 30))
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                xticklabels=column_names, yticklabels=column_names, annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})