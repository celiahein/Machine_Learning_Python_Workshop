#!/bin/bash

module reset
module load python/3.11.5 scipy-stack/2023b

ENV=$HOME/2024-venv
virtualenv --no-download $ENV
source $ENV/bin/activate
pip install --no-index --upgrade pip

# Install various python packages such as scikit-learn and tensorflow
pip install scikit_learn tensorflow --no-index

#pip install -U tensorboard-plugin-profile # Profiler
#pip install seaborn
#pip install xgboost

#Install ipykernel
pip install --no-index ipykernel
python -m ipykernel install --user --name 2024-ml --display-name "2024-ml Kernel"


