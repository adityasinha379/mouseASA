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

# Run Model 3.0
# for num in {0..4}
# do
for use_prior in 1 # {0,1}
do
    for dataset in 'both'
    do
        for BATCH_SIZE in 16
        do
            logfile="logs/test.out"
            # logfile="logs/log_${celltype}.${dataset}.BATCHSIZE${BATCH_SIZE}.out"
            # rm -rf ${logfile}
            bsub -J ${celltype}.${dataset}.${use_prior}.${BATCH_SIZE} -n 20 -R "A100" -W 5:00 -sla llSC2 -q gpuqueue -gpu "num=1" -o ${logfile} -eo ${logfile} "/data/leslie/sinhaa2/mouseASA/scripts/run_m3.sh $dataset $BATCH_SIZE $celltype $poolsize $dropout $use_prior $weight"
        done
        echo "${celltype}.${dataset}.${use_prior}.${BATCH_SIZE}"
    done
done
# done