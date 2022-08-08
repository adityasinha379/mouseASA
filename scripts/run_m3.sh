#!/bin/bash

dataset=$1
BATCH_SIZE=$2
celltype=$3
poolsize=$4
dropout=$5
n_ensemble=$6
use_prior=$7

python /data/leslie/sinhaa2/mouseASA/scripts/run.py $dataset $BATCH_SIZE $celltype $poolsize $dropout $n_ensemble $use_prior
