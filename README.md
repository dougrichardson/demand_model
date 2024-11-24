# demand_model
Codes used for analysis in Richardson et al., Environmental Research Letters, accepted.

The main scripts are:

- 01_Aus_demand: format demand data from Australia Energy Market Operator.
- 02_hourly_to_daily: process hourly ERA5 data to daily
- 03_derived_indices: compute indices to be used as predictors
- 04_population_masks: process population density data
- 05_region_masks: generate masks of population density for states
- 06_predictors: process predictors
- 07_target: process target data
- 08_run_random_forest.py: python script that runs the random forest
- 09_parallel_random_forest.sh: bash script to run multiple models in parallel
- 10_evaluation: prediction skill
- 11_extrapolate_NEM_ERA5: apply the model to all of ERA5
- 12_predictor_exclude: generate data that lies outside of predictor ranges
- 13_analyse_detrended_NEM_ERA5: main analysis of extrapolated data
- functions.py: common functions used throughout
- parallel_inputs: list of model permutations to feed to 09_parallel_random_forest.sh
- environment.yml: yaml file for project

Additional scripts:
- S01_machine_learning_sandpit: notebook used to test machine learning models, eventually formalised in 08_run_random_forest.py
