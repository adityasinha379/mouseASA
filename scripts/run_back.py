import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import h5py
from model import alleleScan, place_tensor, fourier_att_prior_loss
import os
from utils import revcomp_old, subsample_unegs

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_data(celltype, dataset, ident, get_rc=True, frac=0.6):
    basedir = '/data/leslie/shared/ASA/mouseASA/'
    datadir = basedir+'data/'+celltype+'/'
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
        xTr, yTr = revcomp_old(xTr, yTr)
        xVa, yVa = revcomp_old(xVa, yVa)
        xTe, yTe = revcomp_old(xTe, yTe)
    return xTr, xVa, xTe, yTr, yVa, yTe

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        'Initialization'
        self.x = x.swapaxes(1,2)
        self.y = y
        
    def __len__(self):
        'Denotes the total number of samples'
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index].astype(np.float32), self.y[index].astype(np.float32)

def validate(data_loader, model, loss_fcn):
    model.eval()  # Switch to evaluation mode
#     torch.set_grad_enabled(False)
    losses = []
    for input_seqs, output_vals in data_loader:
        input_seqs = input_seqs.to(DEVICE)
        output_vals = output_vals.to(DEVICE)

        logit_pred_vals = model(input_seqs)
        loss = loss_fcn(logit_pred_vals, output_vals)
        losses.append(loss.item())
    return model, losses

def test(data_loader, model):
    model.eval()
    with torch.no_grad():
        preds = []
        for input_seqs,_ in data_loader:
            input_seqs = input_seqs.to(DEVICE)
            preds.append(model(input_seqs).detach().cpu().numpy())
    preds = np.concatenate(preds)
    preds = (preds[:len(preds)//2]+preds[len(preds)//2:])/2        # average of revcomp preds
    # preds = preds[:len(preds)//2]
    return preds

def train(data_loader, model, optimizer, loss_fcn, use_prior, weight=1.0):
    
    model.train()  # Switch to training mode
    losses = []

    for input_seqs, output_vals in data_loader:
        optimizer.zero_grad()
        input_seqs = input_seqs.to(DEVICE)
        output_vals = output_vals.to(DEVICE)
        # Clear gradients from last batch if training

        if use_prior:
            input_seqs.requires_grad = True  # Set gradient required
            logit_pred_vals = model(input_seqs)
            # Compute the gradients of the output with respect to the input
            input_grads, = torch.autograd.grad(
                logit_pred_vals, input_seqs,
                grad_outputs=place_tensor(torch.ones(logit_pred_vals.size())),
                retain_graph=True, create_graph=True)
            # We'll be operating on the gradient itself, so we need to create the graph
            
            input_grads = input_grads * input_seqs  # Gradient * input
            input_seqs.requires_grad = False  # Reset gradient required
            
            fourier_loss = weight*fourier_att_prior_loss(
                    output_vals, input_grads.permute(0,2,1), freq_limit, limit_softness,
                    att_prior_grad_smooth_sigma)
            loss = loss_fcn(logit_pred_vals, output_vals) + fourier_loss
        else:
            logit_pred_vals = model(input_seqs)
            loss = loss_fcn(logit_pred_vals, output_vals)

        loss.backward()  # Compute gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10000)
        optimizer.step()  # Update weights through backprop
        if use_prior:
            losses.append([loss.item()-fourier_loss.item(), fourier_loss.item()])
        else:
            losses.append(loss.item())

    return model, optimizer, losses

def train_model(model, train_loader, valid_loader, num_epochs, optimizer, loss_fcn, SAVEPATH, patience, use_scheduler, use_prior, weight=1.0):
    """
    Trains the model for the given number of epochs.
    """
    best_loss = 1e10
    train_losses = []
    valid_losses = []
    counter = 0

    # torch.autograd.set_detect_anomaly(True)
    for epoch_i in range(num_epochs):
        counter += 1
        model, optimizer, losses = train(train_loader, model, optimizer, loss_fcn, use_prior, weight)
        if use_prior:
            fourier_losses = [x[1] for x in losses]
            losses = [x[0] for x in losses]
            train_loss_mean = np.mean(losses)
            train_fa_loss_mean = np.mean(fourier_losses)
            train_losses.append([train_loss_mean, train_fa_loss_mean])
        else:
            train_loss_mean = np.mean(losses)
            train_losses.append(train_loss_mean)
        
        with torch.no_grad():
            _, losses = validate(valid_loader, model, loss_fcn)
            valid_loss_mean = np.mean(losses)
            valid_losses.append(valid_loss_mean)
            
        if valid_loss_mean < best_loss:
            counter = 0
            print('++++ Val loss improved from {}, saving+++++'.format(best_loss))
            best_loss = valid_loss_mean
            try:
                torch.save(model.state_dict(), SAVEPATH)
            except:
                print('Failed to save.')
        if use_prior:
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                f'Epoch: {epoch_i}\t'
                f'Train loss: {(train_loss_mean+train_fa_loss_mean):.4f} ({train_loss_mean:.4f}+{train_fa_loss_mean:.4f})\t'
                f'Valid loss: {valid_loss_mean:.4f}\t'
            )
        else:
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                f'Epoch: {epoch_i}\t'
                f'Train loss: {train_loss_mean:.4f}\t'
                f'Valid loss: {valid_loss_mean:.4f}\t'
            )
        if counter >= patience:
            print('Val loss did not improve for {} epochs, early stopping...'.format(patience))
            break
        if use_scheduler:    
            scheduler.step()

    return model, train_losses, valid_losses

def get_weight_frac(num):
    if num==0:
        weight, frac = [1.0, 1.0]
    elif num==1:
        weight, frac = [1.0, 0.6]
    elif num==2:
        weight, frac = [1.0, 0.4]
    elif num==3:
        weight, frac = [0.3, 0.6]
    elif num==4:
        weight, frac = [0.7, 0.6]
    elif num==5:
        weight, frac = [1.5, 0.6]
    elif num==6:
        weight, frac = [1.5, 1.0]
    elif num==7:
        weight, frac = [2.0, 1.0]
    elif num==8:
        weight, frac = [2.0, 0.4]
    elif num==9:
        weight, frac = [0.3, 0.4]
    return weight, frac

def get_seed_frac(num):
    if num==0:
        seed, frac = [[0,0], 1.0]
    elif num==1:
        seed, frac = [[0,19], 1.0]
    elif num==2:
        seed, frac = [[0,0], 0.6]
    elif num==3:
        seed, frac = [[0,19], 0.6]
    elif num==4:
        seed, frac = [[19,0], 0.6]
    return seed, frac


if __name__ == "__main__":
    print(DEVICE)
    initial_rate = 1e-3
    wd = 1e-3
    RANDOM_SEED = 0
    N_EPOCHS = 100
    patience = 10

    dataset = sys.argv[1] #'both'
    BATCH_SIZE = int(sys.argv[2]) #32
    celltype = sys.argv[3] #'cd8'
    poolsize = int(sys.argv[4]) #2
    dropout = float(sys.argv[5]) #0.2
    use_prior = 1 #int(sys.argv[6])
    try:
        weight = float(sys.argv[7])  # fourier loss weighting
    except:
        weight = 1.0
    # weight = 1.0
    # num = int(sys.argv[7])
    # print(num)

    # seed, frac = get_seed_frac(num)
    # weight, frac = get_weight_frac(num)

    gc = ''
    ident = '_vi_150bp'
    modelname = 'm3'

    basedir = '/data/leslie/shared/ASA/mouseASA/'
    if use_prior:
        print('fourier_prior')
        #fourier param
        freq_limit = 50
        limit_softness = 0.2
        att_prior_grad_smooth_sigma = 3
    else:
        print('no prior')
    print(modelname)

    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic=True
    np.random.seed(RANDOM_SEED)
    model = alleleScan(poolsize, dropout)
    model.to(DEVICE)

    if modelname=='m3':
        loss_fcn = nn.MSELoss()
    elif modelname=='po':           # still experimenting with this...
        loss_fcn = nn.PoissonNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_rate, weight_decay=wd)

    x_train, x_valid, x_test, y_train, y_valid, y_test = load_data(celltype, dataset, gc+ident, frac=0.6)
    
    ## define the data loaders
    train_dataset = Dataset(x_train, y_train)
    val_dataset = Dataset(x_valid, y_valid)
    test_dataset = Dataset(x_test, y_test)

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
    
    # SAVEPATH = basedir+'ckpt_models/{}/{}_{}_{}_{}_{}{}{}.hdf5'.format(celltype, celltype, modelname, dataset, use_prior, BATCH_SIZE, gc, ident)
    SAVEPATH = basedir+f'ckpt_models/cd8/test_both_m3.hdf5'
    print(SAVEPATH)
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, N_EPOCHS, optimizer, loss_fcn, SAVEPATH, patience, False, use_prior=bool(use_prior), weight=weight)
    model.load_state_dict(torch.load(SAVEPATH))
    
    # run testing with the trained model
    test_preds = test(test_loader, model)     # averaged over revcomps
    predsdir = basedir+'data/'+celltype+'/preds/'
    print(test_preds.shape,'\n')
    if not os.path.exists(predsdir):
        os.makedirs(predsdir)
    # np.save(predsdir+'{}_{}_{}_{}{}{}.npy'.format(modelname, dataset, use_prior, BATCH_SIZE, gc, ident), test_preds)
    np.save(predsdir+f'test_both_m3.npy', test_preds)