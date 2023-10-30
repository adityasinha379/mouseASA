import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from bisect import bisect
from scipy.stats import zscore
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def mutagenize(x):
    # mutagenesis at one base pair
    # x (2, 4)      # b6 and cast nucleotides at a particular position
    mut_x = torch.stack([x]*3)
    acgt = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    for j in range(len(x)):
        mut_x[:,j,:] = torch.tensor(np.delete(acgt, np.where(x[j,:])[0], axis=0), dtype=x.dtype)
    return mut_x      # (3,2,4)

def ISM(x, model):
    # x (2, 4, 300=seqlen)      paired b6 and cast input tensor
    # Code to take a model and an input tensor, perform in-silico saturation mutagenesis and return score matrix
    B = 16
    seqlen = x.shape[-1]      # last dimension of input is seqlen
    # Prepare batched input of ISM perturbations
    x_aug = torch.stack([x]*(3*seqlen+1))                           # 1 for baseline, 3*seqlen for ISM (3*seqlen+1,2,4,300)
    y_aug = np.zeros(x_aug.shape[:-2], dtype=np.float32)                # exclude one-hot and seqlen dims  (3*seqlen+1,2)

    # Prepare ISM input
    for i in range(seqlen):
        x_aug[1+i*3:1+(i+1)*3,:,:,i] = mutagenize(x[:,:,i])
    x_aug = x_aug.to(DEVICE)
    
    # Get outputs
    for i in range(0, len(x_aug), B):    # (B,2,4,300)
        temp = model(x_aug[i:i+B])[0]
        y_aug[i:i+B] = temp.detach().cpu().numpy()    # (B,2)

    # Get pred differences from baseline to calculate score matrix
    temp = (y_aug[1:]-y_aug[0]).reshape(3,-1,2).transpose(2,0,1)     # (2, 3, seqlen)
    # TODO: Do we need to normalize this???
    # Now fill in the 0s at each position based on the original sequence x
    scores = np.zeros((2, 4, seqlen), np.float32)       # define the scores matrix
    for j in range(len(x)):
        for i in range(seqlen):
            scores[j, np.delete([0,1,2,3], np.where(x[j,:,i])[0][0]), i] = temp[j,:,i]
    scores = np.round(scores, 2)
    return scores         # (2, 3, 300)

def trim_weights_dict(weights):
    '''
    LEGACY: Some code I wrote to load weights from a trained pairScan (with FC head) model to a template without the FC head
    Basically, I was having problems doing feature attribution since pairScan gives two outputs and the FA package hates that
    But ultimately, I settled on using a wrapper for my model instead during FA.
    Input: Torch weight dictionary
    Output: Torch weight dictionary
    '''
    temp = []
    for k in weights.keys():       # remove the fchead weights from the loaded weight dictionary
        if k[:6]=='fchead':
            temp.append(k)
    for k in temp:
        del weights[k]
    return weights

def get_neg_summits(chromsummits, num, chrom_length, seed):
    '''
    Code to get "true" uneg summits for a  particular chromosome.
    This samples unegs uniformly from the background instead of from flanks of peaks and generally performs better when compared to taking flanks
    Two caveats about this: 1. We make sure samples are at least 10kb from any positive summit, 2. We don't sample from first and last 5Mb of the chromosome (low mappability areas)
    Input: chromsummits (n,) - list of positive summits
           num (int) - number of samples needed
           chrom_length (int) - length of chromosome
    Output: neg_summits (num,)
    '''
    neg_summits = np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed=seed)
    while True:                # Trial and error, sample summits, take only 10kb separated ones, repeat till you have num samples
        temp = rng.choice(np.arange(5000000, chrom_length-5000000), num, replace=False)
        idx = np.where(np.array([np.min(np.abs(x- chromsummits)) for x in temp])>10000)[0]     # at least 10kb from summit
        neg_summits = np.concatenate((neg_summits, temp[idx]))
        num-=len(idx)
        if num==0:
            break
    return neg_summits

def get_confweights(dataset):
    '''
    DORMANT: We were trying to weight the fold change head loss (fc_loss) with confidence weights, where a peak is higher confidence if it has higher allelic imbalance
    To do this Vianne performed a beta-binomial test to give allelic imbalance pvals to each peak (intuitively, low pval if peak has high FC and high coverage)
    Input: dataset (str) - train, val or test
    Output: zscores (len,) - list of zscores quantifying allelic imbalance for each sample in the dataset
    '''
    if dataset=='train':
        sig_path = '/data/leslie/shared/ASA/mouseASA/data/cd8/betabinom_result_combCounts_150bp_trainOnly.csv'
    elif dataset=='val':
        sig_path = '/data/leslie/shared/ASA/mouseASA/data/cd8/betabinom_result_combCounts_150bp_valOnly.csv'
    pvals = pd.read_csv(sig_path)['p.adj']
    zscores = zscore(-np.log10(pvals+1e-5))
    zscores += abs(min(zscores))
    return zscores

def subsample_unegs(lens, frac=0.5):
    '''
    For when we want to use a variable fraction of unegs for training
    Given list of uneg dataset lengths, returns subsample indices for each
    Input: lens (list of int), frac (float)
    Output: idx (list of list of int)
    '''
    idx = []
    rng = np.random.default_rng(seed=0)
    for i in range(len(lens)): # train, val
        idx.append(np.sort(rng.choice(np.arange(lens[i]), size=int(frac*lens[i]), replace=False)))
    return idx

def get_summits(peaks):
    '''
    LEGACY: Old code I wrote that takes peaks dataframe and gets (positive and) negative summits from the peak flanks (500 bp away from peak edge)
    These negative summits can then be subsampled as you like.
    Could use some overhauling. Eg. probably dont need to return all columns just chr and summit
    As mentioned earlier, random sampling from background seems empirically better, but you can use this to compare
    Input: peaks (pd df) (n,10) loaded from BED file
    Output: summits (n,10) summits_neg (2n,10)
    '''
    # Returns summits corresponding to peaks and also for negative flanking regions
    summits = peaks.copy()
    summits.iloc[:,1] += summits.iloc[:,9]
    summits = summits.iloc[:,[0,1]].reset_index(drop=True)

    summits_neg = summits.copy()
    summits_neg1 = summits.copy()
    for i in range(len(peaks)):
        summits_neg.iloc[i,1] = peaks.iloc[i,2] + 500
        summits_neg1.iloc[i,1] = peaks.iloc[i,1] - 500
    
    summits_neg = pd.concat( (summits_neg,summits_neg1), ignore_index=True)
    summits_neg = summits_neg.sort_values(by=[0,1], ignore_index=True)
    return summits, summits_neg

def get_shifts(chromsummits, mods, c):
    '''
    For a chromosome, get the shift values to convert each summit to cast coordinate
    Input: chromsummits (n,), mods (list n), c (int) chromsome
    Output: cast_shifts (n,)
    '''
    mod_c = [x for x in mods if '\t'+str(c)+'\t' in x]           # slice out relevant mod lines and arrange them in dataframe
    mod_c = pd.DataFrame([x.split('\t') for x in mod_c])
    mod_c[2] = mod_c[2].astype(int)
    idx = [bisect(mod_c[2],summit)-1 for summit in chromsummits]  # get indices corresponding to relevant indels
    
    cast_shifts = []
    for i in idx:
        temp = mod_c.loc[:i]
        cast_shifts.append( len(''.join(temp.loc[temp[0]=='i'][3])) - len(temp.loc[temp[0]=='d'][3]) )
    return cast_shifts

def one_hot(x):
    '''
    Convert input DNA seqs to one-hot
    Input: list n of str
    Output: (n, seqlen, 4)
    '''
    x_oht = []
    mapping = dict(zip(['A','C','G','T','N'],[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]]))   # Order is IMPORTANT
    for i in range(len(x)):
        x_oht.append([mapping[nuc] for nuc in x[i]])
    return np.array(x_oht)

def unhot(x):
    '''
    Convert one-hot input back to str DNA seqs
    Input: x (n, seqlen, 4)
    Output: x_dna (n, seqlen)
    '''
    x_dna = np.full((x.shape[0],x.shape[1]), 'N')
    nucs = ['A','C','G','T']
    for i in range(len(x)):
        for j in range(len(nucs)):
            idx = np.where(x[i,:,j])[0]
            x_dna[i][idx] = nucs[j]
    x_dna = np.array([''.join(x_dna[i]) for i in range(len(x_dna))])
    return x_dna

def GCregress(x, y):
    '''
    Code to perform linear GC regression
    Input: x (n, seqlen, 4),  y: (n,)
    Output: reg_coef (float)
    '''
    GC = np.sum(x, axis=1)
    GC = GC[:,1]+GC[:,2]           # get total GC content for each OHT coded peakseq
    reg = LinearRegression().fit(GC.reshape(-1,1),y.reshape(-1,1))
    return reg.coef_[0][0]

def revcomp_m3(x,y):
    '''
    Augment input dataset with reverse compliments (for alleleScan)
    Input: x (n, seqlen, 4), y (n,)
    Ouput: x (2n, seqlen, 4), y (2n,)
    '''
    x = np.vstack( (x, np.flip(np.flip(x,axis=1),axis=2)) )    # reverse complement the sequences and stack them below already existing peak sequences
    y = np.concatenate((y,y))                    # simply duplicate the log-summed accessibility RPMs
    return x,y

def revcomp(x,y):
    '''
    Augment input dataset with reverse compliments (for pairScan) (TODO: can probably be merged into one function)
    Input: x (n, 2, seqlen, 4), y (n, 2)
    Ouput: x (2n, 2, seqlen, 4), y (2n, 2)
    '''
    x = np.vstack( (x, np.flip(np.flip(x,axis=2),axis=3)) )    # reverse complement the sequences and stack them below already existing peak sequences
    y = np.vstack((y,y))                    # simply duplicate the log-summed accessibility RPMs
    return x,y

def remove_blacklist(peaks, blacklist):
    '''
    LEGACY: Code to filter out the blacklisted peaks
    Input: peaks (n, 10), blacklist (m, 4)
    Output: peaks (n-m, 10)
    '''
    idx = []
    for c in np.unique(peaks[0]):
        chromblack = blacklist.loc[np.where(blacklist[0]=='chr'+str(c))].iloc[:,1:3]
        chromblack = np.array([x for y in chromblack.values.tolist() for x in y])
        chrompeaks = peaks.loc[np.where(peaks[0]==c)]
        temp = chrompeaks.index.tolist()
        for i in range(len(chrompeaks)):
            if np.argmax(chrompeaks.iloc[i,1]<chromblack)%2 or np.argmax(chrompeaks.iloc[i,2]<chromblack)%2:
                idx.append(temp[i])
    peaks = peaks.drop(idx).reset_index(drop=True)     # drop blacklisted peaks
    return peaks


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