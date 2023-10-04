#!/bin/bash
#BSUB -J mouseASA
#BSUB -n 1
#BSUB -R rusage[mem=1]
#BSUB -W 1:00
#BSUB -o logs/log.out
#BSUB -eo logs/log.out

rm -rf logs/log.out

celltype='EpiSC'
strain='pwk'
model_disc='2.2.1'
poolsize=2
dropout='0.2'
weight=1.0
use_prior=1
batch_size=16
fc_frac=''
fc_head='' #_fc for fold change head
model_type='m3'

for dataset in 'both'
do
    for BATCH_SIZE in 16
    do
        logfile="logs/${celltype}_${model_type}_${dataset}_${use_prior}_${batch_size}_${weight}_${model_disc}${fc_head}/log_${celltype}.${strain}.${dataset}.BATCHSIZE${BATCH_SIZE}_${model_disc}${fc_head}.out"
        echo $logfile
        rm -rf ${logfile}
        bsub -J ${celltype}.${strain}.${dataset}.${weight}.${BATCH_SIZE}.${model_disc}.${fc_frac} -n 10 -R "A100" -sla llSC2 -W 10:00 -q gpuqueue -gpu "num=1" -o ${logfile} -eo ${logfile} "/data/leslie/sunge/mouseASA/scripts/run_m3.sh $dataset $BATCH_SIZE $celltype $strain $model_disc $poolsize $dropout $use_prior $fc_frac $weight"
    done
    echo "${celltype}.${dataset}.${weight}.${BATCH_SIZE}"
done