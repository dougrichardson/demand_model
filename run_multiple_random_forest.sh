#!/bin/bash
 
#PBS -P w42
#PBS -q normal
#PBS -l ncpus=1
#PBS -l mem=2GB
#PBS -l jobfs=8GB
#PBS -l walltime=09:00:00
#PBS -l wd
#PBS -l storage=gdata/w42
#PBS -e ./logs/error.txt
#PBS -o ./logs/output.txt

# Run script as: conda run -n <ENV_NAME> python3 run_random_forest.py <PATH> <DEMAND_NETCDF> <MARKET (NEM OR EU)> <REGION e.g. NSW> <REMOVE_WEEKEND BOOL> <REMOVE XMAS BOOL> <REMOVE MONTH INT> <MASK_NAME> <TIME_COLUMNS> <FIRST_TRAIN_YEAR> <LAST_TRAIN_YEAR> <FIRST_TEST_YEAR> <LAST_TEST_YEAR>

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM NEM False False 0 pop_dens_mask is_weekend 2010 2016 2017 2019

# conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM NSW False False 0 pop_dens_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM QLD False False 0 pop_dens_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM SA False False 0 pop_dens_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM VIC False False 0 pop_dens_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM TAS False False 0 pop_dens_mask is_weekend 2010 2016 2017 2019



#### Remove weekend and Xmas

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM NEM True True 0 pop_dens_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM NSW True True 0 pop_dens_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM QLD True True 0 pop_dens_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM SA True True 0 pop_dens_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM VIC True True 0 pop_dens_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM TAS True True 0 pop_dens_mask is_weekend 2010 2016 2017 2019



#### No pop mask
conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM NEM False False 0 no_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM NSW False False 0 no_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM QLD False False 0 no_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM SA False False 0 no_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM VIC False False 0 no_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM TAS False False 0 no_mask is_weekend 2010 2016 2017 2019



#### Log pop mask
conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM NEM False False 0 pop_dens_log_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM NSW False False 0 pop_dens_log_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM QLD False False 0 pop_dens_log_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM SA False False 0 pop_dens_log_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM VIC False False 0 pop_dens_log_mask is_weekend 2010 2016 2017 2019

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM TAS False False 0 pop_dens_log_mask is_weekend 2010 2016 2017 2019