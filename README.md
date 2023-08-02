# mouseASA \
Here's a quick guide introducing you to some of the terminology used in the code and in the documentation of it \
LEGACY: old code I wrote for some reasons that are not relevant anymore, but not deleted just in case (: \
DORMANT: code not currently being used but might be used at a later point \
\
dataset - Term used to talk about the kind of data used for training, accessed from different h5 files that are in turn created by different preprocessing scripts \
    'both' - allele-aware mode, both b6 and cast pseudodiploid data - data_ident.h5 - Compatible with pairScan, alleleScan \
    'trueref' - standard (non-pseudodiploid) aligned data to b6 (baseline) - data_ident_trueref.h5 - Compatible with alleleScan \
    'ref' - pooled pseudodiploid counts with the b6 genome (BAD baseline) - data_ident_ref.h5 - Compatible with pairScan, alleleScan \
    'b6' - allele-specific training with only b6 - data_ident.h5 - Compatible with alleleScan \
    'ca' - allele-specific training with only cast - data_ident.h5 - Compatible with alleleScan \
\
alleleScan takes (n,4,300) input and gives (n,) as output \
pairScan takes (n,2,4,300) input and gives (n,2) as accessibility output and (n,) as fold change output if fc_train is enabled \