import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.ndimage

def place_tensor(tensor):
    """
    Places a tensor on GPU, if PyTorch sees CUDA; otherwise, the returned tensor
    remains on CPU.
    """
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def smooth_tensor_1d(input_tensor, smooth_sigma):
    """
    Smooths an input tensor along a dimension using a Gaussian filter.
    Arguments:
        `input_tensor`: a A x B tensor to smooth along the second dimension
        `smooth_sigma`: width of the Gaussian to use for smoothing; this is the
            standard deviation of the Gaussian to use, and the Gaussian will be
            truncated after 1 sigma (i.e. the smoothing window is
            1 + (2 * sigma); sigma of 0 means no smoothing
    Returns an array the same shape as the input tensor, with the dimension of
    `B` smoothed.
    """
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

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

class resblock(nn.Module):
    def __init__(self,ni):
        super(resblock, self).__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(ni, ni, 3, 1, 1),
            nn.BatchNorm1d(ni),
            nn.ReLU(),
            nn.Conv1d(ni, ni, 1, 1, 0),
            nn.BatchNorm1d(ni),
            nn.ReLU(),
        )

    def forward(self,x):
        residual = x
        out = self.blocks(x)        
        out += residual
        return out

class alleleScan(nn.Module):

    def __init__(self, poolsize, dropout):
        super(alleleScan, self).__init__()
        self.poolsize = poolsize
        self.dropout = dropout
        n = 32
        n2 = 16
        hidden=n
        n_layers=4
        attn_heads=4
        
        self.seq_extractor = nn.Sequential(
            nn.Conv1d(in_channels = 4, out_channels = n2, kernel_size = 15, stride = 1, dilation = 1, padding = 7),
            nn.BatchNorm1d(n2,eps=1e-3),
            nn.ReLU(),
            
            nn.Conv1d(in_channels = n2, out_channels = n, kernel_size = 11, stride = 1, dilation = 1, padding = 5),
            nn.BatchNorm1d(n,eps=1e-3),
            nn.ReLU(),
            
        )
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

        self.seq_extractor_2 = nn.Sequential(
            nn.Conv1d(in_channels = n, out_channels = n2, kernel_size = 5, stride = 1, dilation = 1, padding = 2),
            nn.BatchNorm1d(n2,eps=1e-3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = self.poolsize),  

            nn.Conv1d(in_channels = n2, out_channels = n2, kernel_size = 5, stride = 1, dilation = 1, padding = 2),
            nn.BatchNorm1d(n2,eps=1e-3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = self.poolsize),  
            
        )
        
        self.dense = nn.Sequential(
            nn.Linear(in_features=(1200), out_features=300),
            nn.BatchNorm1d(300,eps=1e-3),
            nn.ReLU(),
            nn.Linear(in_features=(300), out_features=1),
        )


    def forward(self, x):
        x = self.seq_extractor(x)
#         print('1_{}'.format(x.shape))
        x = x.permute(0,2,1)
        
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, None)
#         print('2_{}'.format(x.shape))
        x = x.permute(0,2,1)
#         print('3_{}'.format(x.shape))
        x = self.seq_extractor_2(x)
#         print('3_{}'.format(x.shape))

        x = torch.flatten(x,1)
#         print('flatten_{}'.format(x.shape))
        
        x = self.dense(x)
#         print('dense_{}'.format(x.shape))

        x = torch.flatten(x)
#         print('pred_{}'.format(x.shape))
        return x
    
    def fourier_att_prior_loss(
        self, status, input_grads, freq_limit, limit_softness,
        att_prior_grad_smooth_sigma
    ):
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
        grads_smooth = smooth_tensor_1d(
            abs_grads, att_prior_grad_smooth_sigma
        )

        # Only do the positives
        pos_grads = grads_smooth[status == 1]

        # Loss for positives
        if pos_grads.nelement():
            pos_fft = torch.rfft(pos_grads, 1)
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
                x = place_tensor(
                    torch.arange(1, pos_mags.size(1) - freq_limit + 1)
                ).float()
                weights[:, freq_limit:] = 1 / (1 + torch.pow(x, limit_softness))

            # Multiply frequency magnitudes by weights
            pos_weighted_mags = pos_mags * weights

            # Add up along frequency axis to get score
            pos_score = torch.sum(pos_weighted_mags, dim=1)
            pos_loss = 1 - pos_score
            return torch.mean(pos_loss)
        else:
            return place_tensor(torch.zeros(1))