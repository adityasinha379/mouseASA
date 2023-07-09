import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from bisect import bisect
from scipy.stats import zscore

def trim_weights_dict(weights):
    temp = []
    for k in weights.keys():
        if k[:6]=='fchead':
            temp.append(k)
    for k in temp:
        del weights[k]
    return weights

def get_neg_summits(chromsummits, num, chrom_length):
    neg_summits = np.empty(0, dtype=np.int64)
    while True:
        temp = np.random.choice(np.arange(5000000, chrom_length-5000000), num, replace=False)
        idx = np.where(np.array([np.min(np.abs(x- chromsummits)) for x in temp])>10000)[0]     # at least 10kb from summit
        neg_summits = np.concatenate((neg_summits, temp[idx]))
        num-=len(idx)
        if num==0:
            break
    return neg_summits

def get_confweights(dataset):
    if dataset=='train':
        sig_path = '/data/leslie/shared/ASA/mouseASA/data/cd8/betabinom_result_combCounts_150bp_trainOnly.csv'
    elif dataset=='val':
        sig_path = '/data/leslie/shared/ASA/mouseASA/data/cd8/betabinom_result_combCounts_150bp_valOnly.csv'
    pvals = pd.read_csv(sig_path)['p.adj']
    zscores = zscore(-np.log10(pvals+1e-5))
    zscores += abs(min(zscores))
    return zscores

def subsample_unegs(lens, frac=0.5):
    # for pairScan
    # given list of uneg dataset lengths, returns subsample indices for each
    idx = []
    for i in range(len(lens)): # train, val
        idx.append(np.sort(np.random.choice(np.arange(lens[i]), size=int(frac*lens[i]), replace=False)))
    return idx

def get_summits(peaks):
    # Returns summits corresponding to peaks and also for negative flanking regions
    summits = peaks.copy()
    summits.iloc[:,1] += summits.iloc[:,9]
    summits = summits.iloc[:,[0,1]].reset_index(drop=True)

    summits_neg = summits.copy()
    summits_neg1 = summits.copy()
    for i in range(len(peaks)):
        summits_neg.iloc[i,1] = peaks.iloc[i,2] + 500
        summits_neg1.iloc[i,1] = peaks.iloc[i,1] - (500-1)
    
    summits_neg = pd.concat( (summits_neg,summits_neg1), ignore_index=True)
    summits_neg = summits_neg.sort_values(by=[0,1], ignore_index=True)
    return summits, summits_neg

def get_shifts(chromsummits, mods, c):
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
    # input: list length n of strings
    # output: (n,seqlen,4)
    x_oht = []
    mapping = dict(zip(['A','C','G','T','N'],[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]]))
    for i in range(len(x)):
        x_oht.append([mapping[nuc] for nuc in x[i]])
    return np.array(x_oht)

def GCregress(x, y, method='linear'):
    # x: (n, seqlen, 4)  y: (n,)
    GC = np.sum(x, axis=1)
    GC = GC[:,1]+GC[:,2]           # get total GC content for each OHT coded peakseq
    if method=='linear':
        reg = LinearRegression().fit(GC.reshape(-1,1),y.reshape(-1,1))
        return reg.coef_[0][0]
    elif method=='spline':
        splinereg = get_natural_cubic_spline_model(GC, y, minval=np.min(GC), maxval=np.max(GC), n_knots=6)
        return splinereg.predict(GC)/GC

def revcomp_old(x,y):
    # for alleleScan
    # augment data by adding reverse complements
    # output x: (2n, seqlen, 4)    y: (2n,)
    x = np.vstack( (x, np.flip(np.flip(x,axis=1),axis=2)) )    # reverse complement the sequences and stack them below already existing peak sequences
    y = np.concatenate((y,y))                    # simply duplicate the log-summed accessibility RPMs
    return x,y

def revcomp(x,y):
    # for pairScan
    # augment data by adding reverse complements
    # input x: (n, 2, seqlen, 4)    y: (n, 2)
    # output x: (2n, 2, seqlen, 4)    y: (2n, 2)
    x = np.vstack( (x, np.flip(np.flip(x,axis=2),axis=3)) )    # reverse complement the sequences and stack them below already existing peak sequences
    y = np.vstack((y,y))                    # simply duplicate the log-summed accessibility RPMs
    return x,y

def remove_blacklist(peaks, blacklist):
    idx = []
    for c in np.unique(peaks[0]):
        chromblack = blacklist.loc[np.where(blacklist[0]=='chr'+str(c))].iloc[:,1:3]
        chromblack = np.array([x for y in chromblack.values.tolist() for x in y])
        chrompeaks = peaks.loc[np.where(peaks[0]==c)]
        temp = chrompeaks.index.tolist()
        for i in range(len(chrompeaks)):
            if np.argmax(chrompeaks.iloc[i,1]<chromblack)%2 or np.argmax(chrompeaks.iloc[i,2]<chromblack)%2:
                idx.append(temp[i])
    peaks = peaks.drop(idx).reset_index(drop=True)
    return peaks

def unhot(x):
    # input shape (n,len,4)
    # output shape (n,len)
    x_dna = np.full((x.shape[0],x.shape[1]), 'N')
    nucs = ['A','C','G','T']
    for i in range(len(x)):
        for j in range(len(nucs)):
            idx = np.where(x[i,:,j])[0]
            x_dna[i][idx] = nucs[j]
    x_dna = np.array([''.join(x_dna[i]) for i in range(len(x_dna))])
    return x_dna

def get_natural_cubic_spline_model(x, y, minval=None, maxval=None, n_knots=None, knots=None):
    """
    Get a natural cubic spline model for the data.

    For the knots, give (a) `knots` (as an array) or (b) minval, maxval and n_knots.

    If the knots are not directly specified, the resulting knots are equally
    space within the *interior* of (max, min).  That is, the endpoints are
    *not* included as knots.

    Parameters
    ----------
    x: np.array of float
        The input data
    y: np.array of float
        The outpur data
    minval: float 
        Minimum of interval containing the knots.
    maxval: float 
        Maximum of the interval containing the knots.
    n_knots: positive integer 
        The number of knots to create.
    knots: array or list of floats 
        The knots.

    Returns
    --------
    model: a model object
        The returned model will have following method:
        - predict(x):
            x is a numpy array. This will return the predicted y-values.
    """

    if knots:
        spline = NaturalCubicSpline(knots=knots)
    else:
        spline = NaturalCubicSpline(max=maxval, min=minval, n_knots=n_knots)

    p = Pipeline([
        ('nat_cubic', spline),
        ('regression', LinearRegression(fit_intercept=True))
    ])

    p.fit(x, y)

    return p

class AbstractSpline(BaseEstimator, TransformerMixin):
    """Base class for all spline basis expansions."""

    def __init__(self, max=None, min=None, n_knots=None, n_params=None, knots=None):
        if knots is None:
            if not n_knots:
                n_knots = self._compute_n_knots(n_params)
            knots = np.linspace(min, max, num=(n_knots + 2))[1:-1]
            max, min = np.max(knots), np.min(knots)
        self.knots = np.asarray(knots)

    @property
    def n_knots(self):
        return len(self.knots)

    def fit(self, *args, **kwargs):
        return self

class NaturalCubicSpline(AbstractSpline):
    """Apply a natural cubic basis expansion to an array.
    The features created with this basis expansion can be used to fit a
    piecewise cubic function under the constraint that the fitted curve is
    linear *outside* the range of the knots..  The fitted curve is continuously
    differentiable to the second order at all of the knots.
    This transformer can be created in two ways:
      - By specifying the maximum, minimum, and number of knots.
      - By specifying the cutpoints directly.  

    If the knots are not directly specified, the resulting knots are equally
    space within the *interior* of (max, min).  That is, the endpoints are
    *not* included as knots.
    Parameters
    ----------
    min: float 
        Minimum of interval containing the knots.
    max: float 
        Maximum of the interval containing the knots.
    n_knots: positive integer 
        The number of knots to create.
    knots: array or list of floats 
        The knots.
    """

    def _compute_n_knots(self, n_params):
        return n_params

    @property
    def n_params(self):
        return self.n_knots - 1

    def transform(self, X, **transform_params):
        X_spl = self._transform_array(X)
        if isinstance(X, pd.Series):
            col_names = self._make_names(X)
            X_spl = pd.DataFrame(X_spl, columns=col_names, index=X.index)
        return X_spl

    def _make_names(self, X):
        first_name = "{}_spline_linear".format(X.name)
        rest_names = ["{}_spline_{}".format(X.name, idx)
                      for idx in range(self.n_knots - 2)]
        return [first_name] + rest_names

    def _transform_array(self, X, **transform_params):
        X = X.squeeze()
        try:
            X_spl = np.zeros((X.shape[0], self.n_knots - 1))
        except IndexError: # For arrays with only one element
            X_spl = np.zeros((1, self.n_knots - 1))
        X_spl[:, 0] = X.squeeze()

        def d(knot_idx, x):
            def ppart(t): return np.maximum(0, t)

            def cube(t): return t*t*t
            numerator = (cube(ppart(x - self.knots[knot_idx]))
                         - cube(ppart(x - self.knots[self.n_knots - 1])))
            denominator = self.knots[self.n_knots - 1] - self.knots[knot_idx]
            return numerator / denominator

        for i in range(0, self.n_knots - 2):
            X_spl[:, i+1] = (d(i, X) - d(self.n_knots - 2, X)).squeeze()
        return X_spl

def load_oldcounts():
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
    xTr, yTr = revcomp_old(xTr, yTr)
    xVa, yVa = revcomp_old(xVa, yVa)
    xTe, yTe = revcomp_old(xTe, yTe)
    return xTr, xVa, xTe, yTr, yVa, yTe