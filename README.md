# mouseASA
Here's a quick guide introducing you to some of the terminology used in the code and in the documentation of it \
LEGACY: old code I wrote for some reasons that are not relevant anymore, but not deleted just in case (: \
DORMANT: code not currently being used but might be used at a later point \
\
dataset - Term used to talk about the kind of data used for training, accessed from different h5 files that are in turn created by different preprocessing scripts \
&nbsp; 'both' - allele-aware mode, both b6 and cast pseudodiploid data - data_ident.h5 - Compatible with pairScan, alleleScan \
&nbsp; 'trueref' - standard (non-pseudodiploid) aligned data to b6 (baseline) - data_ident_trueref.h5 - Compatible with alleleScan \
&nbsp; 'ref' - pooled pseudodiploid counts with the b6 genome (BAD baseline) - data_ident_ref.h5 - Compatible with pairScan, alleleScan \
&nbsp; 'b6' - allele-specific training with only b6 - data_ident.h5 - Compatible with alleleScan \
&nbsp; 'ca' - allele-specific training with only cast - data_ident.h5 - Compatible with alleleScan \
\
alleleScan takes (n,4,300) input and gives (n,) as output - ran by run_back.py \
pairScan takes (n,2,4,300) input and gives (n,2) as accessibility output and (n,) as fold change output if fc_train is enabled - run by run.py \
\
Basic job submission: \
The names are a little stupid, I know \
bsub_MouseASA.sh runs the surface level cpu job, which submits one or more gpu jobs in a for loop \
These jobs run run_m3.sh \
run_m3.sh runs either run.py or run_back.py depending on which model you're working with