import xarray as xr
import pandas as pd
import numpy as np
import math
import glob

from workalendar.registry import registry

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
    if first_train_year < last_test_year:
        fy = first_train_year
        ly = last_test_year
    elif first_train_year > last_test_year: # If test period is before training period
        fy = last_test_year
        ly = first_train_year
    else:
        raise ValueError("first training year cannot equal last testing year")
        
    return df[(df.index.year >= fy) & (df.index.year <= ly)]

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
    
    # Difference in standard deviation (of obs and pred)
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    
    # And ratio of standard deviation
    std_ratio = std_pred / std_true
    
    scores.append(cor)
    scores.append(std_ratio)
    
    return scores

def perm_imp(model, X, y, n_repeats, random_state=0):
    """
    Permutation importances
    
    model: sklearn model
    X: array, predictors
    y: array, target
    n_repeats: int, number of iterations
    random_state: int, pseudo random generator
    """
    return permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)

def print_perm_imp(perm_imp, features):
    """
    Print permutation importance stats
    
    perm_imp: output from permutation_importance
    features: list, feature names
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

def get_predictor_files(region, mask, detrended=True):
    """
    Return a list of desired predictor filenames/
    
    region: str, name of region
    mask: str, name of mask
    detrended: bool, whether to load detrended data
    """
    path = "/g/data/w42/dr6273/work/projects/Aus_energy/"
    ext = region + "_" + mask
    if detrended:
        ext = ext + "_detrended"
    return glob.glob(path + "demand_predictors/*" + ext + ".nc")

def to_dataframe(target_da, predictors_ds, region):
    """
    Convert xarray data to pandas dataframe.
    
    target_da: xarray dataArray
    predictors_ds: xarray dataset
    region: str, name of region
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

def add_time_column(df, method, calendar=None):
    """
    Add a time column to df.
    
    df: dataframe to add to
    method: str indicating which method to use. Currently 'is_weekend',
            'month_sin', 'month_cos', 'month_int', 'season_int'
    calendar: None, or calendar from registry
    """
    if method == "is_weekend":
        # Bool for weekend day or weekday
        # new_col = (df.index.weekday > 4).astype("int16")
        
        # Bool for weekend/public holiday or working day
        is_workday = [calendar.is_working_day(pd.to_datetime(i)) for i in df.index.values]
        is_workday = [0 if i else 1 for i in is_workday] # Swap 1s and 0s
        new_col = np.array(is_workday).astype("int16")
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

def get_calendar(market, subregion):
    """
    Get calendar for subregion, using electricity market names
    
    market: str, only "NEM" supported
    subregion: str, region name
    """
    if market == "NEM":
        if subregion == "NEM": # If entire market, use only national holidays
            return registry.get("AU")()
        region = "AU"
    else:
        raise ValueError("Incorrect market specified")
    return registry.get_subregions(region)[region + "-" + subregion]()

def rm_weekend(da, drop=False):
    """
    Set weekend days to NaN
    
    da: array
    drop: bool, whether to drop NaNs
    """
    return da.where(da.time.dt.dayofweek < 5, drop=drop)

def select_workday(da, calendar, drop=False):
    """
    Remove weekends and public holidays
    
    da: array
    calendar: None, or calendar from registry
    drop: bool, whether to drop NaNs
    """
    is_workday = [calendar.is_working_day(pd.to_datetime(i)) for i in da["time"].values]
    da = da.assign_coords({"is_workday": ("time", is_workday)})
    return da.where(da.is_workday == True, drop=drop)

def rm_xmas(da):
    """
    Set 21/12 through 06/01 to NaN
    
    da: array
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

def remove_time(da, weekend=False, xmas=False, month=0, calendar=None):
    """
    Returns da with weekends, xmas, or a month removed if desired
    
    da: array
    weekend: bool, whether to include weekends/public holidays
    xmas: bool, whether to include Christmas period
    calendar: None, or calendar from registry
    """
    if weekend:
        # da = rm_weekend(da)
        da = select_workday(da, calendar)
    if xmas:
        da = rm_xmas(da)
    if month > 0:
        da = rm_month(da, month)
        
    return da.dropna("time")

def get_filename(
    filename, market, region, mask_name,
    first_train_year, last_train_year, first_test_year, last_test_year,
    weekend=False, xmas=False, month=None, nFeatures=None, t_only=False
):
    """
    Return a filename appropriate for the modelling choices made.
    """
    filename = filename + "_" + market + "_" + region + "_" + mask_name
    if weekend:
        filename += "_NOWEEKEND"
    if xmas:
        filename += "_NOXMAS"
    if month > 0:
        filename = filename + "_NOMONTH" + str(month)
        
    filename =  filename + "_training" + str(first_train_year) + "-" + str(last_train_year)
    filename = filename + "_test" + str(first_test_year) + "-" + str(last_test_year)
    
    if nFeatures is not None:
        filename = filename + "_nFeatures-" + nFeatures
        
    if t_only:
        filename = filename + "_t2m_only"
    
    return filename


def read_results(results_name, market, regions, mask_name,
                 first_train_year, last_train_year, first_test_year,
                 last_test_year, rm_weekend, rm_xmas, rm_month,
                 n_features, results_path, detrended=False, t_only=False):
    """
    Read in results dataframes as dictionary items
    """
    if results_name == "feature_selection":
        name = "feature_selection_results"
    elif (results_name == "training") | (results_name == "test"):
        name = results_name + "_predictions"
    else:
        name = results_name
        
    results = dict()
    for r in regions:
        filename = get_filename(
            name, market, r, mask_name,
            first_train_year, last_train_year, first_test_year, last_test_year,
            rm_weekend, rm_xmas, rm_month, n_features, t_only
        )
        if detrended:
            filename = filename + "_detrended"
        results[r] = pd.read_csv(
            results_path + results_name + "/random_forest/" + filename + ".csv",
            index_col=0,
            parse_dates=True
        )
    return results

def sel_model(df):
    """
    Select best model from df
    
    df: dataFrame
    """
    return df.loc[df["selected_features"] == True]

def parse_features(row):
    """
    Parse object to list of strings
    """
    return row.values[0].split("'")[1::2]

# =======================================
# Misc
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

def roundup(x, nearest):
    """
    Round up to nearest integer
    """
    return int(math.ceil(x / nearest)) * nearest

def rounddown(x, nearest):
    """
    Round down to nearest integer
    """
    return int(math.floor(x / nearest)) * nearest

def pretty_variable(var):
    """
    Return 'pretty' variable name
    
    var: str
    """
    if var == "t2m":
        r = r"$T$"
    elif var == "t2m3":
        r = r"$T_{3}$"
    elif var == "t2m4":
        r = r"$T_{4}$"
    elif var == "t2max":
        r = r"$T_{\mathrm{max}}$"
    elif var == "t2min":
        r = r"$T_{\mathrm{min}}$"
    elif var == "cdd":
        r = r"$\mathrm{CDD}$"
    elif var == "cdd3":
        r = r"$\mathrm{CDD_{3}}$"
    elif var == "cdd4":
        r = r"$\mathrm{CDD_{4}}$"
    elif var == "hdd":
        r = r"$\mathrm{HDD}$"
    elif var == "hdd3":
        r = r"$\mathrm{HDD_{3}}$"
    elif var == "hdd4":
        r = r"$\mathrm{HDD_{4}}$"
    elif var == "w10":
        r = r"$W$"
    elif var == "rh":
        r = r"$h$"
    elif var == "q":
        r = r"$q$"
    elif var == "msdwswrf":
        r = r"$R$"
    elif var == "mtpr":
        r = r"$P$"
    else:
        raise ValueError("Incorrect var")
        
    return r