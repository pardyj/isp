#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --account=an-tr043
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=3G
#SBATCH --output=script_output.txt

# add modules
module load python/3.11.5
module load scipy-stack
python -m pip install seaborn
python -m pip install statsmodels
python -m pip install dask
python -m pip install openpyxl
python -m pip install Jinja2

# run python script
python ./script.py

srun hostname
