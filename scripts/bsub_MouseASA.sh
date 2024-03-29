#!/bin/bash
#BSUB -J mouseASA
#BSUB -n 1
#BSUB -R rusage[mem=1]
#BSUB -W 1:00
#BSUB -o logs/log.out
#BSUB -eo logs/log.out

rm -rf logs/log.out

celltype='cd8'
# poolsize=2
# dropout='0.2'
# weight=1.0
# use_prior=1
dataset='both'
modeltype='full'

for BATCH_SIZE in 32
do
    # logfile="logs/log_${celltype}.${dataset}.BATCHSIZE${BATCH_SIZE}.out"
    logfile="logs/test.out"
    # rm -rf ${logfile}
    # bsub -J ${celltype}_${dataset} -n 20 -R "A100" -sla llSC2 -W 5:00 -q gpuqueue -gpu "num=1" -o ${logfile} -eo ${logfile} "/data/leslie/sinhaa2/mouseASA/scripts/run.sh $dataset $BATCH_SIZE $celltype $poolsize $dropout $use_prior $weight"
    bsub -J ${celltype}_${dataset}_${modeltype} -n 20 -R "A100" -sla llSC2 -W 6:00 -q gpuqueue -gpu "num=1" -o ${logfile} -eo ${logfile} "/data/leslie/sinhaa2/mouseASA/scripts/run.sh $dataset $BATCH_SIZE $celltype $modeltype"
done
echo "${celltype}.${dataset}.${BATCH_SIZE}.${modeltype}"