import numpy as np
import torch
import torch.nn as nn


class alleleScan(nn.Module):
    # Indiscriminate architecture
    def __init__(self, poolsize, dropout):
        super(alleleScan, self).__init__()
        self.poolsize = poolsize
        self.dropout = dropout
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

    def forward(self, x):     # (B, 4, 300)
        x = self.seq_extractor(x)
        x = x.permute(0,2,1)   # reshape to make it (n, seq, feat) as required by transformer
        x = self.transformer_block(x)
        x = x.permute(0,2,1)   # reshape back
        x = self.seq_extractor_2(x)
        x = self.seq_extractor_3(x)
        x = torch.flatten(x,1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = torch.flatten(x)
        return x

class pairScan(nn.Module):
    def __init__(self, poolsize, dropout):
        super(pairScan, self).__init__()
        self.poolsize = poolsize
        self.dropout = dropout
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

    def forward(self, x): # (batch, 2, 4, 300)
        x = x.view(-1, x.shape[-2], x.shape[-1])     # (batch*2, 4, 300) stack b6 and cast 
        x = self.seq_extractor(x)
        x = x.permute(2,0,1)    # (seq, batch, feat)
        x = self.transformer_block(x)
        x = x.permute(1,2,0)   # (batch, feat, seq)
        x = self.seq_extractor_2(x)
        x = self.seq_extractor_3(x)
        x = torch.flatten(x,1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = torch.flatten(x)
        x = x.view(-1, 2)          # (batch, 2) separate out for y pred
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
        x = self.backbone(x)      # main pairScan model - Captum hate
        x = x.view(-1)                # (B*2) - Captum love
        return x