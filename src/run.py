from proj1_helpers import *
from implementations import *

DATA_TRAIN_PATH = '../data/train.csv'
y_train, x_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
column_names = np.genfromtxt(DATA_TRAIN_PATH, delimiter=",", dtype=str)[0, 2:]

# FEATURE PROCESSING

#handle invalid values
x_train, column_names = handle_invalid(x_train, column_names)

#remove correlated features
to_remove = ['DER_pt_h', 'DER_sum_pt', 'PRI_met_sumet', 'PRI_jet_all_pt', 'DER_mass_vis']
to_remove = [np.where(column_names == name)[0][0] for name in to_remove]
to_remove = np.array(to_remove)
x_train, column_names = remove_columns(x_train, to_remove), np.delete(column_names, to_remove)

#remove outliers

outlier_row_ids = set()
for i in range(x_train.shape[1]):
    outlier_row_ids.update(detect_outlier(x_train[:, i], 3))
    
x_outlier_free = np.delete(x_train, list(outlier_row_ids), axis=0)
y_outlier_free = np.delete(y_train, list(outlier_row_ids), axis=0)

#Feature expansion

pairwise_poly = tuple([expand_poly(x_outlier_free, i) for i in range(1, 3)])
pairwise_poly = np.hstack(pairwise_poly)
# Check degree performance
for k in range(4, 15):
    col_polynomials = tuple([construct_poly(x_outlier_free, i) for i in range(3, k)])
    col_polynomials = np.hstack(col_polynomials)
    x_expanded = np.hstack((pairwise_poly, col_polynomials))
    x_expanded, _, _ = standardize(x_expanded)
    print('Degree : ', k-1, ' - ', cross_val_acc(y_outlier_free, x_expanded))

col_polynomials = tuple([construct_poly(x_outlier_free, i) for i in range(3, 13)])
col_polynomials = np.hstack(col_polynomials)
x_expanded = np.hstack((pairwise_poly, col_polynomials))


#CREATE MODEL
x_expanded_std, x_exp_mean, x_exp_std = standardize(x_expanded)

ridge_regression_params = {
    "lambda_": [1e-02, 1e-03, 5e-05, 2e-05, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09],
}

best_ridge_params = parameter_grid_search(y_outlier_free, x_expanded_std, ridge_regression, compute_rmse, 
                                    ff_params=ridge_regression_params, verbose=False)[0]["params"]
submit_weights, _ = ridge_regression(y_outlier_free, x_expanded_std, **best_ridge_params)

#Creating submission
DATA_TEST_PATH = '../data/test.csv'
_, x_test, ids_test = load_csv_data(DATA_TEST_PATH)
x_test, _ = handle_invalid(x_test)

# Remove correlated columns
x_test = remove_columns(x_test, to_remove)
# Feature expansion
pairwise_poly = tuple([expand_poly(x_test, i) for i in range(1, 3)])
pairwise_poly = np.hstack(pairwise_poly)

col_polynomials = tuple([construct_poly(x_test, i) for i in range(3, 13)])
col_polynomials = np.hstack(col_polynomials)
x_expanded = np.hstack((pairwise_poly, col_polynomials))

# Standardize
x_test_std, _, _ = standardize(x_expanded, x_exp_mean, x_exp_std)

# Predict labels and create submission
OUTPUT_PATH = '../output/test_pred.csv'
y_pred = predict_labels(submit_weights, x_test_std)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

