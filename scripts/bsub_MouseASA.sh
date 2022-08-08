#!/bin/bash
#BSUB -J MouseASA_test
#BSUB -n 1
#BSUB -R rusage[mem=1]
#BSUB -W 1:00
#BSUB -o logs/log.out
#BSUB -eo logs/log.out

rm -rf logs/log.out

celltype='cd8'
poolsize=2
dropout='0.2'
n_ensemble=3

# Run Model 3.0
for use_prior in 1 # {0,1}
do
    for dataset in {'b6','both'}
    do
        for BATCH_SIZE in 32
        do
            rm -rf logs/log_${celltype}.${dataset}.BATCHSIZE${BATCH_SIZE}.out
            bsub -J ${celltype}.${dataset}.${use_prior}.${BATCH_SIZE} -n 10 -R "A100" -W 3:00 -sla llSC2 -q gpuqueue -gpu "num=1" -o logs/log_${celltype}.${dataset}.BATCHSIZE${BATCH_SIZE}.out -eo logs/log_${celltype}.${dataset}.BATCHSIZE${BATCH_SIZE}.out "/data/leslie/sinhaa2/mouseASA/scripts/run_m3.sh $dataset $BATCH_SIZE $celltype $poolsize $dropout $n_ensemble $use_prior"
        done
        echo "${celltype}.${dataset}.${use_prior}.${BATCH_SIZE}"
    done
done

