import torch
import torch.nn as nn
from torch.distributions.multinomial import Multinomial

def MultinomialNLL(true_counts, logits):
    # Both logits and target have shape (B, outlen) profiles
    logps = nn.LogSoftmax(dim=-1)(logits)
    log_fact_sum = torch.lgamma(torch.sum(true_counts, dim=-1) + 1)   # log(N!) for each sample
    log_prod_fact = torch.sum(torch.lgamma(true_counts + 1), dim=-1) # log(prod_i x_i! i=1 to outlen)
    log_prod_exp = torch.sum(true_counts * logps, dim=-1)
    return torch.mean(-log_fact_sum + log_prod_fact - log_prod_exp)

class ChromBPNet(nn.Module):

    def __init__(self, filters=8, conv_kernel_size=21, profile_kernel_size=75, num_layers=9, seqlen=2114, outlen=1000):
        super(ChromBPNet, self).__init__()
        self.conv_kernel_size = conv_kernel_size
        self.profile_kernel_size = profile_kernel_size
        self.num_layers = num_layers
        self.seqlen = seqlen
        self.filters = filters
        self.outlen = outlen

        self.conv_layers = nn.ModuleList()
        # First non-dilated conv layer
        self.conv_layers.append(nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=self.filters, kernel_size=self.conv_kernel_size, dilation=1, padding='valid'),
            nn.ReLU()))
        
        # Set up dilated conv layers with ReLU activation
        for i in range(1, num_layers):
            self.conv_layers.append( nn.Sequential(
                nn.Conv1d(in_channels=self.filters, out_channels=self.filters, kernel_size=3, dilation=2**i, padding='valid'),
                nn.ReLU())
                )
        
        self.profile_conv = nn.Conv1d(in_channels=self.filters, out_channels=1, kernel_size=self.profile_kernel_size, padding='valid')
        self.dense = nn.Linear(in_features=self.filters, out_features=1)    # for count prediction

    def forward(self, x):  # Input size (batch, 4, seqlen)
        x = self.conv_layers[0](x)
        for i in range(1,self.num_layers):
            conv_x = self.conv_layers[i](x)
            x_len = x.shape[2]
            conv_x_len = conv_x.shape[2]
            assert((x_len - conv_x_len)%2 == 0)     # for symmetric cropping
            crop_size = (x_len - conv_x_len)//2
            x = x[:,:, crop_size:x_len-crop_size]
            x = x + conv_x                          # add the residual after cropping to size

        # Branch 1: Profile Prediction
        prof_precrop = self.profile_conv(x)
        crop_size = (prof_precrop.shape[2] - self.outlen)//2
        assert crop_size>=0
        assert (prof_precrop.shape[2]%2 ==0)
        prof = prof_precrop[:,:, crop_size:prof_precrop.shape[2]-crop_size]
        prof = torch.flatten(prof,1)

        # Branch 2: Counts Prediction
        # Global average pooling along length dimension
        gap = torch.mean(x,2)
        count = torch.flatten(self.dense(gap))

        return prof, count