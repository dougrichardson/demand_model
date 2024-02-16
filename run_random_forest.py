#!/usr/bin/env python
# coding: utf-8

# =================================
# Process demand machine learning
# =================================

# Import libraries
import sys
import os
import glob
import xarray as xr
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import LeaveOneGroupOut

from mlxtend.feature_selection import SequentialFeatureSelector

from scipy.stats import randint, uniform

# Import custom functions from script
os.chdir('/g/data/w42/dr6273/work/demand_model/')
import functions as fn

# Set global variables
PATH = sys.argv[1] # "/g/data/w42/dr6273/work/projects/Aus_energy/"
DEMAND_FILE = sys.argv[2] #"daily_demand_2010-2020_stl.nc"
MARKET = sys.argv[3] #"NEM" # "NEM" or "EU"
REGION = sys.argv[4] #"NSW"
REMOVE_WEEKEND = sys.argv[5] #False
REMOVE_XMAS = sys.argv[6] #False
REMOVE_MONTH = int(sys.argv[7]) #0 # integer: [1, 12]
MASK_NAME = sys.argv[8] #"pop_dens_mask"
TIME_COLUMNS = sys.argv[9] #["is_weekend"]
# Alternatives:
# time_cols = ["is_weekend","month_sin", "month_cos"]
# time_cols = ["is_weekend","month_int"]
# time_cols = ["is_weekend","season_int"]
# time_cols = ["is_weekend","is_transition"]
# time_cols = None
FIRST_TRAIN_YEAR = int(sys.argv[10]) #2010
LAST_TRAIN_YEAR = int(sys.argv[11]) #2011
FIRST_TEST_YEAR = int(sys.argv[12]) #2012
LAST_TEST_YEAR = int(sys.argv[13]) #2012
N_FEATURES = sys.argv[14] # "best" or "parsimonious". For feature selection

# Convert str to required type
# Str to bool
if REMOVE_WEEKEND == "True":
    REMOVE_WEEKEND = True
else:
    REMOVE_WEEKEND = False
    
if REMOVE_XMAS == "True":
    REMOVE_XMAS = True
else:
    REMOVE_XMAS = False

# Str to list
if TIME_COLUMNS == "None":
    TIME_COLUMNS = []
else:
    TIME_COLUMNS = TIME_COLUMNS.split(",")

# # Prepare demand data
# def remove_time(da, weekend=False, xmas=False, month=0):
#     """
#     Returns da with weekends, xmas, or a month removed if desired
#     """
#     if REMOVE_WEEKEND:
#         da = fn.rm_weekend(da)
#     if REMOVE_XMAS:
#         da = fn.rm_xmas(da)
#     if REMOVE_MONTH > 0:
#         da = fn.rm_month(da, REMOVE_MONTH)
        
#     return da.dropna("time")

dem_da = xr.open_dataset(PATH + "data/energy_demand/" + DEMAND_FILE)["demand_stl"]
dem_da = fn.remove_time(dem_da, REMOVE_WEEKEND, REMOVE_XMAS, REMOVE_MONTH)
dem_da = dem_da.sel(region=REGION).expand_dims({"region": [REGION]})

# Prepare predictors
files = fn.get_predictor_files(MARKET, MASK_NAME)
pred_ds = xr.open_mfdataset(files, combine="nested", compat="override")
pred_ds = pred_ds.sel(region=REGION).expand_dims({"region": [REGION]}).compute()
pred_ds = fn.remove_time(pred_ds, REMOVE_WEEKEND, REMOVE_XMAS, REMOVE_MONTH)

# Prepare dataframe for machine learning
region_dfs = {}
for region in dem_da.region.values:
    df = fn.to_dataframe(dem_da, pred_ds, region)
    for t in TIME_COLUMNS:
        df = fn.add_time_column(df, t)
    new_cols = np.append(np.append("demand", TIME_COLUMNS), df.columns[:-(len(TIME_COLUMNS) + 1)])
    df = df[new_cols]
    region_dfs[region] = df
# region_dfs[REGION] = region_dfs[REGION][["demand", "hdd", "cdd"]]

# Split data into training and testing

test_len = dem_da.sel(time=slice(str(FIRST_TEST_YEAR), str(LAST_TEST_YEAR))).time.values.shape[0]

train_X, test_X, train_y, test_y = fn.split(
    fn.sel_train_test(region_dfs[REGION], FIRST_TRAIN_YEAR, LAST_TEST_YEAR),
    "demand",
    test_size=test_len,
    random_state=0,
    shuffle=False
)

# Sequential feature selection
rf = ExtraTreesRegressor(
    random_state=0
)

logo = fn.leave_one_group_out(
    train_X,
    train_y,
    dem_da.sel(time=slice(str(FIRST_TRAIN_YEAR), str(LAST_TRAIN_YEAR))),
    str(FIRST_TRAIN_YEAR),
    str(LAST_TRAIN_YEAR)
)

model = fn.mlextend_sfs(
    train_X,
    train_y,
    rf,
    list(logo),
    True,
    scoring="neg_mean_absolute_error",
    k_features=N_FEATURES
)

features = region_dfs[region].columns[1:]
selected_features = list(features[list(model.k_feature_idx_)])

results_df = pd.DataFrame.from_dict(model.get_metric_dict()).T

feature_names = [
    [features[i] for i in results_df["feature_idx"].iloc[j]]
    for j in range(len(results_df))
]

results_df["feature_names"] = feature_names

# Boolean column indicating which model was selected (useful for reading this dataframe back later)
results_df["selected_features"] = [len(i) == len(selected_features) for i in results_df["feature_idx"]]

# Write results to file
filename = fn.get_filename(
    "feature_selection_results", MARKET, REGION, MASK_NAME,
    FIRST_TRAIN_YEAR, LAST_TRAIN_YEAR, FIRST_TEST_YEAR, LAST_TEST_YEAR,
    REMOVE_WEEKEND, REMOVE_XMAS, REMOVE_MONTH, N_FEATURES
)

results_df.to_csv(
    PATH + "model_results/feature_selection/random_forest/" + filename + ".csv",
)

#  Tune hyperparameters
# Using leave one group out cross validation, where a group is a year.
parameters = {
    "n_estimators": randint(200, 500), # no. trees in the forest
    "min_samples_leaf": randint(5, 30), # min no. samples at leaf node
    "max_depth" : randint(5, 50), # max depth of each tree
    "max_leaf_nodes": randint(20, 200) # size of tree, how many end nodes
}

# ========================= !!!!!!!!!!!!!!!!!!!!!!  very restricted space for TESTING
# parameters = {
#     "n_estimators": randint(20, 22), # no. trees in the forest
#     "min_samples_leaf": randint(5, 7), # min no. samples at leaf node
#     "max_depth" : randint(5, 7), # max depth of each tree
#     "max_leaf_nodes": randint(20, 22) # size of tree, how many end nodes
# }
# ========================= !!!!!!!!!!!!!!!!!!!!!!

retain = ["demand"] + selected_features
final_features = region_dfs[region][retain]

train_X, test_X, train_y, test_y = fn.split(
    fn.sel_train_test(final_features, FIRST_TRAIN_YEAR, LAST_TEST_YEAR),
    "demand",
    test_size=test_len,
    random_state=0,
    shuffle=False
)

rf = ExtraTreesRegressor()

logo = fn.leave_one_group_out(
    train_X,
    train_y,
    dem_da.sel(time=slice(str(FIRST_TRAIN_YEAR), str(LAST_TRAIN_YEAR))),
    str(FIRST_TRAIN_YEAR),
    str(LAST_TRAIN_YEAR)
)

# !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE n_iter for actual runs
# ========================= !!!!!!!!!!!!!!!!!!!!!!
# ========================= !!!!!!!!!!!!!!!!!!!!!!
best_params = fn.tune_hyperparameters(train_X, train_y, rf, parameters, logo, n_iter=200)
best_params_df = pd.Series(
    [best_params[i] for i in list(best_params.keys())],
    index=list(best_params.keys())
)

filename = fn.get_filename(
    "hyperparameters", MARKET, REGION, MASK_NAME,
    FIRST_TRAIN_YEAR, LAST_TRAIN_YEAR, FIRST_TEST_YEAR, LAST_TEST_YEAR,
    REMOVE_WEEKEND, REMOVE_XMAS, REMOVE_MONTH, N_FEATURES
)
best_params_df.to_csv(
    PATH + "model_results/hyperparameters/random_forest/" + filename + ".csv",
)

# Finalise model
rf = ExtraTreesRegressor(
    n_estimators=best_params["n_estimators"],
    min_samples_leaf=best_params["min_samples_leaf"],
    max_depth=best_params["max_depth"],
    max_leaf_nodes=best_params["max_leaf_nodes"],
    random_state=0,
)

model_train, pred_train = fn.predict_forest(train_y, train_X, train_X, rf)
model_test, pred_test = fn.predict_forest(train_y, train_X, test_X, rf)

# Dataframes of observations and predictions for training and testing period
# Training
train_df = pd.DataFrame(
    np.vstack([train_y, pred_train]).transpose(),
    columns=["observation", "prediction"],
    index=dem_da.sel(time=slice(str(FIRST_TRAIN_YEAR), str(LAST_TRAIN_YEAR))).time
)
filename = fn.get_filename(
    "training_predictions", MARKET, REGION, MASK_NAME,
    FIRST_TRAIN_YEAR, LAST_TRAIN_YEAR, FIRST_TEST_YEAR, LAST_TEST_YEAR,
    REMOVE_WEEKEND, REMOVE_XMAS, REMOVE_MONTH, N_FEATURES
)
train_df.to_csv(
    PATH + "model_results/training/random_forest/" + filename + ".csv",
)

# Test
test_df = pd.DataFrame(
    np.vstack([test_y, pred_test]).transpose(),
    columns=["observation", "prediction"],
    index=dem_da.sel(time=slice(str(FIRST_TEST_YEAR), str(LAST_TEST_YEAR))).time
)
filename = fn.get_filename(
    "test_predictions", MARKET, REGION, MASK_NAME,
    FIRST_TRAIN_YEAR, LAST_TRAIN_YEAR, FIRST_TEST_YEAR, LAST_TEST_YEAR,
    REMOVE_WEEKEND, REMOVE_XMAS, REMOVE_MONTH, N_FEATURES
)
test_df.to_csv(
    PATH + "model_results/test/random_forest/" + filename + ".csv",
)