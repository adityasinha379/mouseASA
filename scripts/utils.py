import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from bisect import bisect
from scipy.stats import zscore
import scipy.ndimage
import torch
from sklearn.metrics import precision_recall_curve, roc_curve, auc

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def place_tensor(tensor):
    '''
    Places a tensor on GPU if PyTorch sees CUDA, else returned tensor remains on CPU.
    '''
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def smooth_tensor_1d(input_tensor, smooth_sigma):
    '''
    For fourier prior
    Smooths an input tensor along a dimension using a Gaussian filter.
    Arguments:
        `input_tensor`: a A x B tensor to smooth along the second dimension
        `smooth_sigma`: width of the Gaussian to use for smoothing; this is the
            standard deviation of the Gaussian to use, and the Gaussian will be
            truncated after 1 sigma (i.e. the smoothing window is
            1 + (2 * sigma); sigma of 0 means no smoothing
    Returns an array the same shape as the input tensor, with the dimension of
    `B` smoothed.
    '''
    # Generate the kernel
    if smooth_sigma == 0:
        sigma, truncate = 1, 0
    else:
        sigma, truncate = smooth_sigma, 1
    base = np.zeros(1 + (2 * sigma))
    base[sigma] = 1  # Center of window is 1 everywhere else is 0
    kernel = scipy.ndimage.gaussian_filter(base, sigma=sigma, truncate=truncate)
    kernel = place_tensor(torch.tensor(kernel))

    # Expand the input and kernel to 3D, with channels of 1
    # Also make the kernel float-type, as the input is going to be of type float
    input_tensor = torch.unsqueeze(input_tensor, dim=1)
    kernel = torch.unsqueeze(torch.unsqueeze(kernel, dim=0), dim=1).float()

    smoothed = torch.nn.functional.conv1d(
        input_tensor, kernel, padding=sigma
    )

    return torch.squeeze(smoothed, dim=1)

def fourier_att_prior_loss(status, input_grads, freq_limit, limit_softness, att_prior_grad_smooth_sigma):
    """
    Computes an attribution prior loss for some given training examples,
    using a Fourier transform form.
    Arguments:
        `status`: a B-tensor, where B is the batch size; each entry is 1 if
            that example is to be treated as a positive example, and 0
            otherwise
        `input_grads`: a B x L x 4 tensor, where B is the batch size, L is
            the length of the input; this needs to be the gradients of the
            input with respect to the output; this should be
            *gradient times input*
        `freq_limit`: the maximum integer frequency index, k, to consider for
            the loss; this corresponds to a frequency cut-off of pi * k / L;
            k should be less than L / 2
        `limit_softness`: amount to soften the limit by, using a hill
            function; None means no softness
        `att_prior_grad_smooth_sigma`: amount to smooth the gradient before
            computing the loss
    Returns a single scalar Tensor consisting of the attribution loss for
    the batch.
    """
    abs_grads = torch.sum(torch.abs(input_grads), dim=2)
    # Smooth the gradients
    grads_smooth = smooth_tensor_1d(abs_grads, att_prior_grad_smooth_sigma)
    # Only do the positives
    pos_grads = grads_smooth[status > -1.]
    
    # Loss for positives
    if pos_grads.nelement():
        pos_fft = torch.view_as_real(torch.fft.rfft(pos_grads, dim=1))
        pos_mags = torch.norm(pos_fft, dim=2)
        pos_mag_sum = torch.sum(pos_mags, dim=1, keepdim=True)
        pos_mag_sum[pos_mag_sum == 0] = 1  # Keep 0s when the sum is 0
        pos_mags = pos_mags / pos_mag_sum
        # Cut off DC
        pos_mags = pos_mags[:, 1:]
        # Construct weight vector
        weights = place_tensor(torch.ones_like(pos_mags))
        if limit_softness is None:
            weights[:, freq_limit:] = 0
        else:
            x = place_tensor(torch.arange(1, pos_mags.size(1) - freq_limit + 1)).float()
            weights[:, freq_limit:] = 1 / (1 + torch.pow(x, limit_softness))

        # Multiply frequency magnitudes by weights
        pos_weighted_mags = pos_mags * weights
        # Add up along frequency axis to get score
        pos_score = torch.sum(pos_weighted_mags, dim=1)
        pos_loss = 1 - pos_score
        return torch.mean(pos_loss)
    else:
        return place_tensor(torch.zeros(1))

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
    rng = np.random.default_rng(seed=0)
    idx = rng.choice(np.arange(len(summits_neg)), len(summits_neg)//2, replace=False)
    summits_neg = summits_neg.iloc[idx, :].reset_index(drop=True)
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

def plot_auc(x,y,name,ax):
    if name=='roc':
        roc = roc_curve(x,y)
        ax.plot(roc[0],roc[1])
        ax.set_ylabel('TPR')
        ax.set_xlabel('FPR')
        ax.text(0.2, 0.9, 'AUC = {:.3f}'.format(auc(roc[0],roc[1])), ha='center', va='bottom',fontsize=12)
    elif name=='prc':
        prc = precision_recall_curve(x,y)
        ax.plot(prc[1],prc[0])
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        ax.text(0.8, 0.9, 'AUPRC = {:.3f}'.format(auc(prc[1],prc[0])), ha='center', va='bottom',fontsize=12)
    return