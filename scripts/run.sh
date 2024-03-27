#!/bin/bash

# dataset=$1
# BATCH_SIZE=$2
# celltype=$3
# poolsize=$4
# dropout=$5
# use_prior=$6
# weight=$7

# python /data/leslie/sinhaa2/mouseASA/scripts/run_pairscan.py $dataset $BATCH_SIZE $celltype $poolsize $dropout $use_prior $weight


dataset=$1
BATCH_SIZE=$2
celltype=$3
modeltype=$4

python /data/leslie/sinhaa2/mouseASA/scripts/chrombpnet/run_chrombpnet.py $dataset $BATCH_SIZE $celltype $modeltype
