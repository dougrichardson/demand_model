import xarray as xr
import pandas as pd
import numpy as np
import glob

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import LeaveOneGroupOut

from sklearn.feature_selection import SequentialFeatureSelector

from mlxtend.feature_selection import SequentialFeatureSelector

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, median_absolute_error

from sklearn.inspection import permutation_importance

from scipy.stats import randint, uniform, pearsonr

# =======================================
# Machine learning
# =======================================

def sel_train_test(df, first_train_year, last_test_year):
    """
    Selects df for training and test years. Might be used to e.g.
    exclude a validation set.
    
    df: dataframe
    first_train_year, last_test_year: int, first year of training set and last year of test set
    """
    return df[(df.index.year >= first_train_year) & (df.index.year <= last_test_year)]

def split(df, target_name, test_size, random_state, shuffle=True):
    """
    Apply train_test_split to dataframe
    
    df: pandas dataframe
    target_name: column name of df to target
    test_size: float, proportion of data to test
    random_state: int
    shuffle: bool, whether or not to shuffle data before splitting
    """
    y = np.array(df[target_name]) # target
    X = np.array(df.drop(target_name, axis=1)) # predictors
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

def sfs(train_X, train_y, model, cv, direction="forward", tol=0.001, scoring="r2"):
    """
    Forward selection of features using CV, and a chosen score with tolerance.
    
    train_X: training data
    train_y: training target
    model: sklearn model e.g. RandomForestRegressor
    cv: the cross validation to do. See docs for RandomizedSearchCV
    direction: str, forward or backward selection
    tol: tolerance for scoring
    scoring: score to use in CV
    """
    s = SequentialFeatureSelector(model, direction=direction, tol=tol, scoring=scoring, cv=cv)
    return s.fit(train_X, train_y)

def mlextend_sfs(train_X, train_y, rf, cv, forward, scoring, k_features="best"):
    """
    mlextend Forward selection of features using CV, and a chosen score with tolerance.
    
    train_X: training data
    train_y: training target
    rf: RandomForestRegressor
    cv: the cross validation to do. See docs for RandomizedSearchCV
    forward: bool, forward or backward selection
    tol: tolerance for scoring
    scoring: score to use in CV
    """
    s = SequentialFeatureSelector(rf, k_features=k_features, forward=forward, scoring=scoring, cv=cv)
    return s.fit(train_X, train_y)

def leave_one_group_out(train_X, train_y, target_da, first_year, last_year):
    """
    Return CV splitter using leave one group out, where the groups are years.
    
    train_X: training data
    train_y: training target
    target_da: target array with time dimension
    first_year, last_year: str, first and last years of training set
    """
    logo = LeaveOneGroupOut()
    groups = target_da.sel(time=slice(first_year, last_year)).time.dt.year.values
    return logo.split(train_X, train_y, groups=groups)

def tune_hyperparameters(
    train_X,
    train_y,
    model,
    parameters,
    cv,
    n_iter,
    scoring="neg_mean_absolute_error",
    random_state=0,
    verbose=0
):
    """
    Return tuned hyperparameters by fitting the model using CV on the training set.
    
    train_X: training data
    train_y: training target
    model: sklearn model e.g. RandomForestRegressor
    parameters: dict, parameters of RandomForestRegressor to tune, with associated distributions
    cv: the cross validation to do. See docs for RandomizedSearchCV
    scoring: scoring to use in RandomizedSearchCV
    random_state: random state for RandomizedSearchCV
    n_iter: number of CV iterations
    verbose: int, amount of data to print
    """
    clf = RandomizedSearchCV(
        model,
        parameters,
        cv=cv,
        n_iter=n_iter,
        scoring=scoring,
        random_state=random_state,
        verbose=verbose
    )
    search = clf.fit(train_X, train_y)
    return search.best_params_

def predict_forest(train_y, train_X, test_X, model):
    """
    Instantiate and fit a random forest model, return predictions.
    
    train_y: target for training set
    train_X: predictors for training set
    test_X: predictors for test set
    model: sklearn model e.g. RandomForestRegressor
    """
    model.fit(train_X, train_y)
    return model, model.predict(test_X)

def compute_scores(y_true, y_pred, metrics):
    """
    Compute scores for a variety of metrics.
    
    y_true: observations
    y_predict: predictions
    metrics: list of scores from sklearn.metrics
    """
    scores = []
    for metric in metrics:
        s = metric(y_true, y_pred)
        scores.append(s)
    # Also compute Pearson correlation
    cor = pearsonr(y_true, y_pred)[0]
    scores.append(cor)
    return scores

def perm_imp(model, X, y, n_repeats, random_state=0):
    """
    Permutation importances
    """
    return permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)

def print_perm_imp(perm_imp, features):
    """
    Print permutation importance stats
    """
    for i in perm_imp.importances_mean.argsort()[::-1]:
        if perm_imp.importances_mean[i] - 2 * perm_imp.importances_std[i] > 0:
            print(
                f"{features[i]:<8} "
                f"{perm_imp.importances_mean[i]:.3f}"
                f" +/- {perm_imp.importances_std[i]:.3f}"
            )

# =======================================
# Reading and preparing data
# =======================================

def get_predictor_files(region, mask):
    """
    Return a list of desired predictor filenames/
    
    region: str, name of region
    mask: str, name of mask
    """
    path = "/g/data/w42/dr6273/work/projects/Aus_energy/"
    ext = region + "_" + mask
    return glob.glob(path + "demand_predictors/*" + ext + "*")

def to_dataframe(target_da, predictors_ds, region):
    """
    Convert xarray data to pandas dataframe.
    """
    # Predictors to array
    predictors_arr = predictors_ds.sel(region=region, time=target_da["time"]).to_array("variable")
    
    # Data array of target and predictors
    da = xr.concat([
        predictors_arr,
        target_da.sel(region=region).expand_dims({"variable": ["demand"]})
    ],
        dim="variable"
    )
    
    # Dataframe
    df = pd.DataFrame(
        da.transpose(),
        columns=da["variable"],
        index=target_da["time"]
    )
    
    return df

def add_time_column(df, method):
    """
    Add a time column to df.
    
    df: dataframe to add to
    method: str indicating which method to use. Currently 'is_weekend',
            'month_sin', 'month_cos', 'month_int', 'season_int'
    """
    if method == "is_weekend":
        # Bool for weekend day or weekday
        new_col = (df.index.weekday > 4).astype("int16")
    elif method == "is_transition":
        new_col = df.index.month.isin([3, 4, 5, 9, 10, 11])
    elif method == "month_sin":
        new_col = np.sin((df.index.month - 1) * (2. * np.pi / 12))
    elif method == "month_cos":
        new_col = np.cos((df.index.month - 1) * (2. * np.pi / 12))
    elif method == "month_int":
        new_col = df.index.month
    elif method == "season_int":
        new_col = df.index.month % 12 // 3 + 1
    else:
        raise ValueError("Incorrect 'method'.")
    
    df[method] = new_col
    return df

def rm_weekend(da):
    """
    Set weekend days to NaN
    """
    return da.where(da.time.dt.dayofweek < 5, drop=False)

def rm_xmas(da):
    """
    Set 21/12 through 06/01 to NaN
    """
    da_ = da.where(
        da.where(
            (da.time.dt.month == 12) & 
            (da.time.dt.day > 20)
        ).isnull()
    )
    da_ = da_.where(
        da_.where(
            (da_.time.dt.month == 1) & 
            (da_.time.dt.day < 7)
        ).isnull()
    )
    return da_

def rm_month(da, month):
    """
    Set a particular month to NaN
    """
    if month not in range(1, 13):
        raise ValueError("Month must be integer between 1 and 12")
    return da.where(da.time.dt.month != month)

# =======================================
# Data wrangling
# =======================================

def detrend_dim(da, dim, deg=1):
    """
    Detrend along a single dimension.
    
    da: array to detrend
    dim: dimension along which to detrend
    deg: degree of polynomial to fit (1 for linear fit)
    
    Adapted from the original code here:
    Author: Ryan Abernathy
    From: https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
    """
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit