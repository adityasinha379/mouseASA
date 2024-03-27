import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
from model import ChromBPNet, MultinomialNLL
from loaders import load_data_chrombpnet
from training import train_model, test

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, p):
        'Initialization'
        self.x = x.swapaxes(1,2)
        self.y = y
        self.p = p
        
    def __len__(self):
        'Denotes the total number of samples'
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index].astype(np.float32), self.y[index].astype(np.float32), self.p[index].astype(np.float32)


if __name__ == "__main__":
    initial_rate = 1e-3
    wd = 1e-2
    N_epochs = 2
    patience = 10

    dataset = sys.argv[1] #'both'
    BATCH_SIZE = int(sys.argv[2]) #32
    celltype = sys.argv[3] #'cd8'
    modeltype = sys.argv[4] # 'bias' or 'full'

    ident = '_vi_chrom_aug'
    modelname = 'ch'

    basedir = f'/data/leslie/shared/ASA/mouseASA/{celltype}/cast'
    print(modelname, modeltype)

    # Load either bias training data or actual augmented data
    x_tr, x_va, x_te, y_tr, y_va, y_te, p_tr, p_va, p_te = load_data_chrombpnet(celltype, dataset, ident, modeltype)
    
    ## define the data loaders
    train_dataset = Dataset(x_tr, y_tr, p_tr)
    val_dataset = Dataset(x_va, y_va, p_va)
    test_dataset = Dataset(x_te, y_te, p_te)

    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True,
                            num_workers =1)
    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=False,
                        num_workers = 1)
    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=False,
                        num_workers = 1)
    
    torch.backends.cudnn.deterministic=True
    loss_fcn = [nn.MSELoss(), MultinomialNLL]     # (y, p)
    
    if not os.path.exists(f'{basedir}/ckpt_models/'):
        os.makedirs(f'{basedir}/ckpt_models/')

    if modeltype=='bias':
        ident+='_bias'
        SAVEPATH = f'{basedir}/ckpt_models/{modelname}_{dataset}_{BATCH_SIZE}_{ident}.hdf5'
        model = ChromBPNet(filters=8, conv_kernel_size=21, profile_kernel_size=75, num_layers=9, seqlen=2114, outlen=1000)
        model.to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_rate, weight_decay=wd)
        # Train bias model
        model, train_losses, val_losses = train_model(model, train_loader, val_loader, N_epochs, optimizer, loss_fcn, SAVEPATH, patience, modeltype)
        model.load_state_dict(torch.load(SAVEPATH))
        test_preds_y, test_preds_p = test(test_loader, model)
        
    elif modeltype=='full':
        SAVEPATH = f'{basedir}/ckpt_models/{modelname}_{dataset}_{BATCH_SIZE}_{ident}_bias.hdf5'
        model_bias = ChromBPNet(filters=8, conv_kernel_size=21, profile_kernel_size=75, num_layers=9, seqlen=2114, outlen=1000)
        model_bias.to(DEVICE)
        model_bias.load_state_dict(torch.load(SAVEPATH))
        for p in model_bias.parameters():          # Freeze the bias model
            p.requires_grad = False
        model_bias.eval()
        temp = test(test_loader, model_bias)   # for sanity check
        
        SAVEPATH = f'{basedir}/ckpt_models/{modelname}_{dataset}_{BATCH_SIZE}_{ident}.hdf5'
        model = ChromBPNet(filters=8, conv_kernel_size=21, profile_kernel_size=75, num_layers=9, seqlen=2114, outlen=1000)
        model.load_state_dict(torch.load(SAVEPATH))
        model.to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_rate, weight_decay=wd)     # Optimize only over the non-bias model
        # Train non-bias model
        model, train_losses, val_losses = train_model([model_bias, model], train_loader, val_loader, N_epochs, optimizer, loss_fcn, SAVEPATH, patience, modeltype)
        model_bias = model[0]
        model = model[1]
        model.load_state_dict(torch.load(SAVEPATH))
        test_preds_y, test_preds_p = test(test_loader, model)
        temp1 = test(test_loader, model_bias) # sanity check
        for i in range(len(temp)):
            print(np.allclose(temp[i], temp1[i]))
    
    
    predsdir = f'{basedir}/preds/'
    print(test_preds_y.shape)
    print(test_preds_p.shape,'\n')
    if not os.path.exists(predsdir):
        os.makedirs(predsdir)
    np.savez(f'{predsdir}/{modelname}_{dataset}_{BATCH_SIZE}_{ident}.npy', y=test_preds_y, p=test_preds_p)