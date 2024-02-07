import numpy as np
import h5py
import pandas as pd
from utils import revcomp, revcomp_m3, subsample_unegs, one_hot


def load_data_pairscan(celltype, dataset, ident, get_rc=True, frac=0.3, finetune=False):
    basedir = f'/data/leslie/shared/ASA/mouseASA/{celltype}/cast/data/'

    with h5py.File(basedir+'data'+ident+'.h5','r') as f:
        if dataset=='both' and not finetune:         # for normal allele aware training
            uneg_idx = subsample_unegs([len(f['x_train_b6_unegs'][()]), len(f['x_val_b6_unegs'][()])], frac=frac)
            xTr = np.stack(( np.vstack((f['x_train_b6'][()], f['x_train_b6_unegs'][()][uneg_idx[0]])),
                np.vstack((f['x_train_cast'][()], f['x_train_cast_unegs'][()][uneg_idx[0]])) ), axis=1)      # (n, 2, 300, 4)
            yTr = np.stack(( np.concatenate((f['y_train_b6'][()], f['y_train_unegs'][()][uneg_idx[0]])),
                np.concatenate((f['y_train_cast'][()], f['y_train_unegs'][()][uneg_idx[0]])) ), axis=-1)  # (n, 2)
            xVa = np.stack(( np.vstack((f['x_val_b6'][()], f['x_val_b6_unegs'][()][uneg_idx[1]])),
            np.vstack((f['x_val_cast'][()], f['x_val_cast_unegs'][()][uneg_idx[1]])) ), axis=1)
            yVa = np.stack(( np.concatenate((f['y_val_b6'][()], f['y_val_unegs'][()][uneg_idx[1]])),
                np.concatenate((f['y_val_cast'][()], f['y_val_unegs'][()][uneg_idx[1]])) ), axis=-1)
        elif dataset=='both' and finetune:           # for fine tuning pretrained model
            idx_tr = np.where(pd.read_csv(basedir+'significance/betabinom_result_combCounts_150bp_trainOnly.csv')['p.adj'] < 0.05)[0]
            idx_va = np.where(pd.read_csv(basedir+'significance/betabinom_result_combCounts_150bp_valOnly.csv')['p.adj'] < 0.05)[0]
            xTr = np.stack(( f['x_train_b6'][()][idx_tr], f['x_train_cast'][()][idx_tr] ), axis=1)      # (n, 2, 300, 4)
            yTr = np.stack(( f['y_train_b6'][()][idx_tr], f['y_train_cast'][()][idx_tr] ), axis=-1)  # (n, 2)
            xVa = np.stack(( f['x_val_b6'][()][idx_va], f['x_val_cast'][()][idx_va] ), axis=1)
            yVa = np.stack(( f['y_val_b6'][()][idx_va], f['y_val_cast'][()][idx_va] ), axis=-1)
        xTe = np.stack((f['x_test_b6'][()],f['x_test_cast'][()]), axis=1)
        yTe = np.stack((f['y_test_b6'][()],f['y_test_cast'][()]), axis=-1)

    if dataset=='ref':               # for pretraining reference model
        with h5py.File(basedir+'data'+ident+'_'+dataset+'.h5','r') as f:
            uneg_idx = subsample_unegs([len(f['x_train_unegs'][()]), len(f['x_val_unegs'][()])], frac=frac)
            xTr = np.vstack((f['x_train'][()], f['x_train_unegs'][()][uneg_idx[0]]))      # (n, 300, 4)
            yTr = np.concatenate((f['y_train'][()], f['y_train_unegs'][()][uneg_idx[0]]))  # (n, )
            xVa = np.vstack((f['x_val'][()], f['x_val_unegs'][()][uneg_idx[1]]))
            yVa = np.concatenate((f['y_val'][()], f['y_val_unegs'][()][uneg_idx[1]]))
            if get_rc:
                xTr, yTr = revcomp_m3(xTr, yTr)
                xVa, yVa = revcomp_m3(xVa, yVa)
                xTe, yTe = revcomp(xTe, yTe)
            xTr = xTr.reshape(-1, 2, xTr.shape[-2], xTr.shape[-1])          # (n, 2, 300, 4)  (get it into pairScan format)
            xVa = xVa.reshape(-1, 2, xVa.shape[-2], xVa.shape[-1])
            yTr = yTr.reshape(-1,2)       # (n, 2)  (get it into pairScan format)
            yVa = yVa.reshape(-1,2)

    # Augment each dataset with revcomps, test for pred averaging
    if dataset=='both' and get_rc:
        xTr, yTr = revcomp(xTr, yTr)
        xVa, yVa = revcomp(xVa, yVa)
        xTe, yTe = revcomp(xTe, yTe)
    return xTr, xVa, xTe, yTr, yVa, yTe


def load_data_allelescan(celltype, dataset, ident, get_rc=True, frac=0.6):
    basedir = '/data/leslie/shared/ASA/mouseASA/'
    datadir = basedir+celltype+'/cast/data/'
    
    if dataset=='trueref' or dataset=='ref':
        with h5py.File(datadir+'data'+ident+'_'+dataset+'.h5','r') as f:
            uneg_idx = subsample_unegs([len(f['x_train_unegs'][()]), len(f['x_val_unegs'][()])], frac=frac)
            xTr = np.vstack((f['x_train'][()], f['x_train_unegs'][()][uneg_idx[0]]))
            yTr = np.concatenate((f['y_train'][()], f['y_train_unegs'][()][uneg_idx[0]]))
            xVa = np.vstack((f['x_val'][()], f['x_val_unegs'][()][uneg_idx[1]]))
            yVa = np.concatenate((f['y_val'][()], f['y_val_unegs'][()][uneg_idx[1]]))

    with h5py.File(datadir+'data'+ident+'.h5','r') as f:
        uneg_idx = subsample_unegs([len(f['x_train_b6_unegs'][()]), len(f['x_val_b6_unegs'][()])], frac=frac)
        if dataset=='both':
            xTr = np.vstack((f['x_train_b6'][()], f['x_train_cast'][()], f['x_train_b6_unegs'][()][uneg_idx[0]]))
            yTr = np.concatenate((f['y_train_b6'][()], f['y_train_cast'][()], f['y_train_unegs'][()][uneg_idx[0]]))
            xVa = np.vstack((f['x_val_b6'][()], f['x_val_cast'][()], f['x_val_b6_unegs'][()][uneg_idx[1]]))
            yVa = np.concatenate((f['y_val_b6'][()], f['y_val_cast'][()], f['y_val_unegs'][()][uneg_idx[1]]))
        elif dataset=='b6':
            xTr = np.vstack((f['x_train_b6'][()], f['x_train_b6_unegs'][()][uneg_idx[0]]))
            yTr = np.concatenate((f['y_train_b6'][()], f['y_train_unegs'][()][uneg_idx[0]]))
            xVa = np.vstack((f['x_val_b6'][()], f['x_val_b6_unegs'][()][uneg_idx[1]]))
            yVa = np.concatenate((f['y_val_b6'][()], f['y_val_unegs'][()][uneg_idx[1]]))
        elif dataset=='ca':
            xTr = np.vstack((f['x_train_cast'][()], f['x_train_cast_unegs'][()][uneg_idx[0]]))
            yTr = np.concatenate((f['y_train_cast'][()], f['y_train_unegs'][()][uneg_idx[0]]))
            xVa = np.vstack((f['x_val_cast'][()], f['x_val_cast_unegs'][()][uneg_idx[1]]))
            yVa = np.concatenate((f['y_val_cast'][()], f['y_val_unegs'][()][uneg_idx[1]]))
        # Test dataset remains same
        xTe = np.vstack((f['x_test_b6'][()],f['x_test_cast'][()]))
        yTe = np.concatenate((f['y_test_b6'][()],f['y_test_cast'][()]))
    
    # Augment each dataset with revcomps, test for pred averaging
    if get_rc:
        xTr, yTr = revcomp_m3(xTr, yTr)
        xVa, yVa = revcomp_m3(xVa, yVa)
        xTe, yTe = revcomp_m3(xTe, yTe)
    return xTr, xVa, xTe, yTr, yVa, yTe

def load_oldcounts():
    '''
    LEGACY: Code to load some old counts generated by Vianne for comparison
    Will never be used by you or me for that matter. Ignore this.
    '''
    datadir = '/data/leslie/shared/ASA/mouseASA/data/cd8/'
    df = pd.read_csv(datadir+'data_vi_old.csv')
    x_b6 = one_hot(df['b6_seq'])
    x_ca = one_hot(df['ca_seq'])
    y_b6 = df['b6_total_log21p_count_gc_regressed']
    y_ca = df['ca_total_log21p_count_gc_regressed']
    tr_idx = df['chr'].isin([1,2,4,6,7,8,9,11,12,13,14,16,17,18,19])
    va_idx = df['chr'].isin([3,5])
    te_idx = df['chr'].isin([10,15])
    xTr = np.vstack((x_b6[tr_idx], x_ca[tr_idx]))
    xVa = np.vstack((x_b6[va_idx], x_ca[va_idx]))
    xTe = np.vstack((x_b6[te_idx], x_ca[te_idx]))
    yTr = np.concatenate((y_b6[tr_idx], y_ca[tr_idx]))
    yVa = np.concatenate((y_b6[va_idx], y_ca[va_idx]))
    yTe = np.concatenate((y_b6[te_idx], y_ca[te_idx]))
    xTr, yTr = revcomp_m3(xTr, yTr)
    xVa, yVa = revcomp_m3(xVa, yVa)
    xTe, yTe = revcomp_m3(xTe, yTe)
    return xTr, xVa, xTe, yTr, yVa, yTe