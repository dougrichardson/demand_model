#!/bin/bash
 
#PBS -P w42
#PBS -q normal
#PBS -l ncpus=1
#PBS -l mem=2GB
#PBS -l jobfs=8GB
#PBS -l walltime=00:05:00
#PBS -l wd
#PBS -l storage=gdata/w42
 
# # Load module, always specify version number.
# module load python
# python -m venv --system-site-packages /g/data/w42/dr6273/apps/conda/envs/ml_env
#conda activate pangeo_ML

# # Run Python applications
#python run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM NSW False False 0 pop_dens_mask is_weekend 2010 2011 2012 2012
#> $PBS_JOBID.log

conda run -n pangeo_ML python3 run_random_forest.py /g/data/w42/dr6273/work/projects/Aus_energy/ daily_demand_2010-2020_stl.nc NEM NSW False False 0 pop_dens_mask is_weekend 2010 2011 2012 2012

#echo "TEST"
 
# # Deactivate virtual environment, if any.
#conda deactivate
