#!/bin/bash

dataset=$1
BATCH_SIZE=$2
celltype=$3
strain=$4
model_disc=$5
poolsize=$6
dropout=$7
use_prior=$8
fc_frac=$9
weight=${10}

python /data/leslie/sunge/mouseASA/scripts/run_back.py $dataset $BATCH_SIZE $celltype $strain $model_disc $poolsize $dropout $use_prior $fc_frac $weight