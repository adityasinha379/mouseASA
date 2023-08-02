#!/bin/bash
#BSUB -J mouseASA
#BSUB -n 1
#BSUB -R rusage[mem=1]
#BSUB -W 1:00
#BSUB -o logs/log.out
#BSUB -eo logs/log.out

rm -rf logs/log.out

celltype='cd8'
poolsize=2
dropout='0.2'
weight=1.0
use_prior=1


for dataset in 'both'
do
    for BATCH_SIZE in 16
    do
        logfile="logs/log_${celltype}.${dataset}.BATCHSIZE${BATCH_SIZE}.out"
        rm -rf ${logfile}
        bsub -J ${celltype}.${dataset}.${weight}.${BATCH_SIZE} -n 10 -R "A100" -sla llSC2 -W 3:00 -q gpuqueue -gpu "num=1" -o ${logfile} -eo ${logfile} "/data/leslie/sinhaa2/mouseASA/scripts/run_m3.sh $dataset $BATCH_SIZE $celltype $poolsize $dropout $use_prior $weight"
    done
    echo "${celltype}.${dataset}.${weight}.${BATCH_SIZE}"
done