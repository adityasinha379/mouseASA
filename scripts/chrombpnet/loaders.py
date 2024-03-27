import numpy as np
import h5py

def load_data_chrombpnet(celltype, dataset, ident, modeltype, get_rc=True):
    basedir = '/data/leslie/shared/ASA/mouseASA/'
    datadir = basedir+celltype+'/cast/data/'
    
    chrom_train = [1,2,4,6,7,8,9,11,12,13,14,16,17,18,19]
    chrom_val = [3,5]
    chrom_test = [10,15]

    if dataset=='both':
        with h5py.File(datadir+'data'+ident+'.h5','r') as f:
            if modeltype=='bias':
                xTr = np.vstack([f[k] for k in f.keys() for c in chrom_train if 'x_chr'+str(c)+'_' in k and 'unegs' in k])
                xVa = np.vstack([f[k] for k in f.keys() for c in chrom_val if 'x_chr'+str(c)+'_' in k and 'unegs' in k])
                xTe = np.vstack([f[k] for k in f.keys() for c in chrom_test if 'x_chr'+str(c)+'_' in k and 'unegs' in k])
                pTr = np.vstack([f[k] for k in f.keys() for c in chrom_train if 'p_chr'+str(c)+'_' in k and 'unegs' in k])
                pVa = np.vstack([f[k] for k in f.keys() for c in chrom_val if 'p_chr'+str(c)+'_' in k and 'unegs' in k])
                pTe = np.vstack([f[k] for k in f.keys() for c in chrom_test if 'p_chr'+str(c)+'_' in k and 'unegs' in k])
            
            elif modeltype=='full':
                xTr = np.vstack([f[k] for k in f.keys() for c in chrom_train if 'x_chr'+str(c)+'_' in k and 'unegs' not in k])
                xVa = np.vstack([f[k] for k in f.keys() for c in chrom_val if 'x_chr'+str(c)+'_' in k and 'unegs' not in k])
                xTe = np.vstack([f[k] for k in f.keys() for c in chrom_test if 'x_chr'+str(c)+'_' in k and 'unegs' not in k])
                pTr = np.vstack([f[k] for k in f.keys() for c in chrom_train if 'p_chr'+str(c)+'_' in k and 'unegs' not in k])
                pVa = np.vstack([f[k] for k in f.keys() for c in chrom_val if 'p_chr'+str(c)+'_' in k and 'unegs' not in k])
                pTe = np.vstack([f[k] for k in f.keys() for c in chrom_test if 'p_chr'+str(c)+'_' in k and 'unegs' not in k])
            
            yTr = np.log(1+np.sum(pTr, axis=1))
            yVa = np.log(1+np.sum(pVa, axis=1))
            yTe = np.log(1+np.sum(pTe, axis=1))
    
    # Augment each dataset with revcomps, test for pred averaging
    if get_rc:
        xTr, yTr, pTr = revcomp_ch(xTr, yTr, pTr)
        xVa, yVa, pVa = revcomp_ch(xVa, yVa, pVa)
        # xTe, yTe, pTe = revcomp_ch(xTe, yTe, pTe)   # NOTE: no revcomp averaging so no need for revcomps
    return xTr, xVa, xTe, yTr, yVa, yTe, pTr, pVa, pTe

def revcomp_ch(x, y, p):
    '''
    Augment input dataset with reverse compliments (for alleleScan)
    Input: x (n, seqlen, 4), y (n,), p (n, seqlen)
    Ouput: x (2n, seqlen, 4), y (2n,), p (2n, seqlen)
    '''
    x = np.vstack( (x, np.flip(np.flip(x,axis=1),axis=2)) )    # reverse complement the sequences and stack them below already existing peak sequences
    y = np.concatenate((y,y))                    # simply duplicate the log-summed accessibility RPMs
    p = np.vstack( (p, np.flip(p, axis=1)) )             # flip the profiles left to right
    return x, y, p