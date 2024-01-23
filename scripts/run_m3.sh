#!/bin/bash

dataset=$1
BATCH_SIZE=$2
celltype=$3
poolsize=$4
dropout=$5
use_prior=$6
weight=$7

python /data/leslie/sinhaa2/mouseASA/scripts/run_back.py $dataset $BATCH_SIZE $celltype $poolsize $dropout $use_prior $weight
