#!/bin/bash

module reset
module load python/3.11.5 scipy-stack/2023b

ENV=$HOME/2024-venv

source $ENV/bin/activate

# Optional to start Jupyter Notebook server
#salloc --time=2:0:0 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=2G --account=cossw19-wa --reservation cossw19-wr_cpu srun $VIRTUAL_ENV/bin/notebook.sh

