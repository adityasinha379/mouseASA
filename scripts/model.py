import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.ndimage

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


class alleleScan(nn.Module):
    # Indiscriminate architecture
    def __init__(self, poolsize, dropout):
        super(alleleScan, self).__init__()
        self.poolsize = poolsize        # for max pooling at the end
        self.dropout = dropout
        n = 32
        n2 = 16
        n_layers=2
        attn_heads=4
        
        self.seq_extractor = nn.Sequential(
            nn.Conv1d(in_channels = 4, out_channels = n2, kernel_size = 15, stride = 1, dilation = 1, padding = 7),
            nn.BatchNorm1d(n2,eps=1e-3),
            nn.ReLU(),
            
            nn.Conv1d(in_channels = n2, out_channels = n, kernel_size = 11, stride = 1, dilation = 1, padding = 5),
            nn.BatchNorm1d(n,eps=1e-3),
            nn.ReLU())
        
        self.transformer_blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=n, nhead=attn_heads, dim_feedforward=n*4, dropout=self.dropout, batch_first=True), num_layers=n_layers)

        self.seq_extractor_2 = nn.Sequential(
            nn.Conv1d(in_channels = n, out_channels = n2, kernel_size = 5, stride = 1, dilation = 1, padding = 2),
            nn.BatchNorm1d(n2,eps=1e-3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = self.poolsize),  

            nn.Conv1d(in_channels = n2, out_channels = n2, kernel_size = 5, stride = 1, dilation = 1, padding = 2),
            nn.BatchNorm1d(n2,eps=1e-3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = self.poolsize))
        
        self.dense = nn.Sequential(
            nn.Linear(in_features=(1200), out_features=300),
            nn.BatchNorm1d(300,eps=1e-3),
            nn.ReLU(),
            nn.Linear(in_features=(300), out_features=1))

    def forward(self, x):     # (B, 4, 300)
        x = self.seq_extractor(x)
        x = x.permute(0,2,1)   # reshape to make it (n, seq, feat) as required by transformer
        x = self.transformer_blocks(x)
        x = x.permute(0,2,1)   # reshape back
        x = self.seq_extractor_2(x)
        x = torch.flatten(x,1)
        x = self.dense(x)
        x = torch.flatten(x)
        return x


def fc_loss(output, target, conf_weights=1.):\
    # MSE weighted by confidence weights of fold change. If not provided, defaults to standard MSE
    loss = torch.mean(conf_weights*(output-target)**2)
    return loss

class fcblock(nn.Module):
    def __init__(self,n):
        super(fcblock, self).__init__()
        self.final = nn.Linear(in_features=(n), out_features=1)

    def forward(self,x1,x2):
        x = torch.cat([x1,x2], dim=1)
        out = torch.flatten( self.final(x) )
        return out

class pairScan(nn.Module):
    def __init__(self, poolsize, dropout, fc_train=True):
        super(pairScan, self).__init__()
        self.poolsize = poolsize
        self.dropout = dropout
        self.fc_train = fc_train
        n = 32
        n2 = 16
        n3 = 8
        n_layers=2
        attn_heads=4
        
        self.seq_extractor = nn.Sequential(
            nn.Conv1d(in_channels = 4, out_channels = n2, kernel_size = 15, stride = 1, dilation = 1, padding = 'same'),
            nn.BatchNorm1d(n2,eps=1e-3),
            nn.ReLU(),
            
            nn.Conv1d(in_channels = n2, out_channels = n, kernel_size = 11, stride = 1, dilation = 1, padding = 'same'),
            nn.BatchNorm1d(n,eps=1e-3),
            nn.ReLU())
        
        self.transformer_block = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=n, nhead=attn_heads, dim_feedforward=n*4, dropout=self.dropout, layer_norm_eps=1e-3), num_layers=n_layers)
        
        self.seq_extractor_2 = nn.Sequential(
            nn.Conv1d(in_channels = n, out_channels = n3, kernel_size = 5, stride = 1, dilation = 1, padding = 'same'),
            nn.BatchNorm1d(n3,eps=1e-3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = self.poolsize))
        self.seq_extractor_3 = nn.Sequential(
            nn.Conv1d(in_channels = n3, out_channels = 4, kernel_size = 5, stride = 1, dilation = 1, padding = 'same'),
            nn.BatchNorm1d(4,eps=1e-3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = self.poolsize))
        
        self.dense1 = nn.Sequential(
            nn.Linear(in_features=(300), out_features=100),
            nn.BatchNorm1d(100,eps=1e-3),
            nn.ReLU())
        self.dense2 = nn.Linear(in_features=(100), out_features=1)
        
        if self.fc_train:
            self.fchead = fcblock(200)

    def forward(self, x): # (batch, 2, 4, 300)
        x = x.view(-1, x.shape[-2], x.shape[-1])     # (batch*2, 4, 300) stack b6 and cast 
        x = self.seq_extractor(x)
        x = x.permute(2,0,1)    # (seq, batch, feat)
        x = x.permute(1,2,0)   # (batch, feat, seq)
        x = self.seq_extractor_2(x)
        x = self.seq_extractor_3(x)
        x = torch.flatten(x,1)
        x = self.dense1(x)
        if self.fc_train:
            h = x.clone().view(-1, 2, x.shape[-1]).permute(1,0,2)   # (batch, 2, d)
        x = self.dense2(x)
        x = torch.flatten(x)
        x = x.view(-1, 2)          # (batch, 2) separate out for y pred
        if self.fc_train:
            fc = self.fchead(h[0],h[1])
            return x, fc
        else:
            return x

class pairScanWrapper(nn.Module):
    '''
    Captum Integrated Gradients hates multiple outputs, and is strict with input and output shapes
    This is not a problem for alleleScan, but for pairScan we have to write a wrapper that calls the pairScan backbone
    See analysis_pairScan.ipynb FA section for more details
    '''
    def __init__(self, model_obj):
        super(pairScanWrapper, self).__init__()
        self.backbone = model_obj     # model with trained weights
    
    def forward(self, x): # (B*2, 4, 300) - Captum love
        x = x.view(-1, 2, x.shape[-2], x.shape[-1])  # (B, 2, 4, 300) - Captum hate
        x,_ = self.backbone(x)      # main pairScan model - Captum hate
        x = x.view(-1)                # (B*2) - Captum love
        return x