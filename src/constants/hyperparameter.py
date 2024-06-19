""" Hyperparameters"""

RIDGE_PARAMS = {
    "alpha": [0.1, 0.5, 1.0, 5.0, 10.0],
    "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"],
}

LASSO_PARAMS = {"alpha": [0.1, 0.5, 1.0, 5.0, 10.0], "selection": ["cyclic", "random"]}

POLYNOMIAL_PARAMS = {"polynomialfeatures__degree": [2, 3, 4]}

RANDOM_FOREST_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "max_features": ["auto", "sqrt", "log2"],
}

GRADIENTBOOST_PARAMS = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7],
}

"""Single valued hyperparameters"""

XGB_PARAMS = {
    "verbosity": 0,
    "colsample_bytree": 0.3,
    "learning_rate": 0.7,
    "max_depth": 15,
    "alpha": 1,
    "n_estimators": 100,
}

LGB_PARAMS = {
    "verbose": -1,
    "objective": "regression",  # specify the objective for regression
    "metric": "mse",  # evaluation metric
    "verbosity": -1,  # suppress output
    "boosting_type": "gbdt",  # gradient boosting decision tree
    "learning_rate": 0.1,
    "num_leaves": 31,  # maximum number of leaves in one tree
    "max_depth": -1,  # no limit on tree depth
    "n_estimators": 100,
}

CATBOOST_PARAMS = {
    "silent": True,
    "loss_function": "RMSE",  # specify the loss function for regression
    "iterations": 100,  # number of boosting iterations
    "learning_rate": 0.1,
    "depth": 16,  # depth of the trees
    "random_seed": 42,
}
