import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
import sys
import h5py
from model import pairScan, place_tensor, fourier_att_prior_loss
import os
from utils import revcomp, revcomp_m3, subsample_unegs

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 0

def load_data(celltype, dataset, ident, get_rc=True, frac=0.3, finetune=False):
    basedir = f'/data/leslie/shared/ASA/mouseASA/{celltype}/cast/data/'

    with h5py.File(basedir+'data'+ident+'.h5','r') as f:
        if dataset=='both' and not finetune:         # for normal allele aware training
            uneg_idx = subsample_unegs([len(f['x_train_b6_unegs'][()]), len(f['x_val_b6_unegs'][()])], frac=frac)
            xTr = np.stack(( np.vstack((f['x_train_b6'][()], f['x_train_b6_unegs'][()][uneg_idx[0]])),
                np.vstack((f['x_train_cast'][()], f['x_train_cast_unegs'][()][uneg_idx[0]])) ), axis=1)      # (n, 2, 300, 4)
            yTr = np.stack(( np.concatenate((f['y_train_b6'][()], f['y_train_unegs'][()][uneg_idx[0]])),
                np.concatenate((f['y_train_cast'][()], f['y_train_unegs'][()][uneg_idx[0]])) ), axis=-1)  # (n, 2)
            xVa = np.stack(( np.vstack((f['x_val_b6'][()], f['x_val_b6_unegs'][()][uneg_idx[1]])),
            np.vstack((f['x_val_cast'][()], f['x_val_cast_unegs'][()][uneg_idx[1]])) ), axis=1)
            yVa = np.stack(( np.concatenate((f['y_val_b6'][()], f['y_val_unegs'][()][uneg_idx[1]])),
                np.concatenate((f['y_val_cast'][()], f['y_val_unegs'][()][uneg_idx[1]])) ), axis=-1)
        elif dataset=='both' and finetune:           # for fine tuning pretrained model
            idx_tr = np.where(pd.read_csv(basedir+'significance/betabinom_result_combCounts_150bp_trainOnly.csv')['p.adj'] < 0.05)[0]
            idx_va = np.where(pd.read_csv(basedir+'significance/betabinom_result_combCounts_150bp_valOnly.csv')['p.adj'] < 0.05)[0]
            xTr = np.stack(( f['x_train_b6'][()][idx_tr], f['x_train_cast'][()][idx_tr] ), axis=1)      # (n, 2, 300, 4)
            yTr = np.stack(( f['y_train_b6'][()][idx_tr], f['y_train_cast'][()][idx_tr] ), axis=-1)  # (n, 2)
            xVa = np.stack(( f['x_val_b6'][()][idx_va], f['x_val_cast'][()][idx_va] ), axis=1)
            yVa = np.stack(( f['y_val_b6'][()][idx_va], f['y_val_cast'][()][idx_va] ), axis=-1)
        xTe = np.stack((f['x_test_b6'][()],f['x_test_cast'][()]), axis=1)
        yTe = np.stack((f['y_test_b6'][()],f['y_test_cast'][()]), axis=-1)

    if dataset=='ref':               # for pretraining reference model
        with h5py.File(basedir+'data'+ident+'_'+dataset+'.h5','r') as f:
            uneg_idx = subsample_unegs([len(f['x_train_unegs'][()]), len(f['x_val_unegs'][()])], frac=frac)
            xTr = np.vstack((f['x_train'][()], f['x_train_unegs'][()][uneg_idx[0]]))      # (n, 300, 4)
            yTr = np.concatenate((f['y_train'][()], f['y_train_unegs'][()][uneg_idx[0]]))  # (n, )
            xVa = np.vstack((f['x_val'][()], f['x_val_unegs'][()][uneg_idx[1]]))
            yVa = np.concatenate((f['y_val'][()], f['y_val_unegs'][()][uneg_idx[1]]))
            if get_rc:
                xTr, yTr = revcomp_m3(xTr, yTr)
                xVa, yVa = revcomp_m3(xVa, yVa)
                xTe, yTe = revcomp(xTe, yTe)
            xTr = xTr.reshape(-1, 2, xTr.shape[-2], xTr.shape[-1])          # (n, 2, 300, 4)  (get it into pairScan format)
            xVa = xVa.reshape(-1, 2, xVa.shape[-2], xVa.shape[-1])
            yTr = yTr.reshape(-1,2)       # (n, 2)  (get it into pairScan format)
            yVa = yVa.reshape(-1,2)

    # Augment each dataset with revcomps, test for pred averaging
    if dataset=='both' and get_rc:
        xTr, yTr = revcomp(xTr, yTr)
        xVa, yVa = revcomp(xVa, yVa)
        xTe, yTe = revcomp(xTe, yTe)
    return xTr, xVa, xTe, yTr, yVa, yTe

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        'Initialization'
        self.x = x.swapaxes(2,3)  # (batch, 2, 4, 300)
        self.y = y
        self.fc = y[:,1] - y[:,0]
        
    def __len__(self):
        'Denotes the total number of samples'
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index].astype(np.float32), self.y[index].astype(np.float32), self.fc[index].astype(np.float32)


def test(data_loader, model):
    model.eval()
    with torch.no_grad():
        preds = []
        for input_seqs,_,_ in data_loader:
            input_seqs = input_seqs.to(DEVICE)
            logit_pred_vals = model(input_seqs)
            preds.append(logit_pred_vals.detach().cpu().numpy())
    preds = np.concatenate(preds)
    preds = ((preds[:len(preds)//2]+preds[len(preds)//2:])/2).T.reshape(-1)        # average of revcomp preds
    # preds = preds[:len(preds)//2]
    return preds

def validate(data_loader, model, loss_fcn):
    model.eval()  # Switch to evaluation mode
    losses = []
    for input_seqs, output_vals, fc in data_loader:
        input_seqs = input_seqs.to(DEVICE)
        output_vals = output_vals.to(DEVICE)
        fc = fc.to(DEVICE)

        logit_pred_vals = model(input_seqs)
        loss = loss_fcn(logit_pred_vals, output_vals)
        losses.append(loss.item())
    return model, losses

def train(data_loader, model, optimizer, loss_fcn, use_prior, weight=1.0):
    model.train()  # Switch to training mode
    losses = []
    cnt=0
    for input_seqs, output_vals, fc in data_loader:
        cnt+=1
        optimizer.zero_grad()
        input_seqs = input_seqs.to(DEVICE)
        output_vals = output_vals.to(DEVICE)
        fc = fc.to(DEVICE)
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
            fourier_loss = weight*fourier_att_prior_loss(output_vals.reshape(-1),
                input_grads.permute(0,1,3,2).reshape(-1,input_grads.shape[-1],input_grads.shape[-2]),
                freq_limit, limit_softness, att_prior_grad_smooth_sigma)
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

def train_model(model, train_loader, valid_loader, num_epochs, optimizer, loss_fcn, SAVEPATH, patience, use_prior, weight=1.0):
    """
    Trains the model for the given number of epochs.
    """
    best_loss = 1e10
    train_losses = []
    valid_losses = []
    counter = 0
    ## SWA
    # swa_model = AveragedModel(model)

    for epoch_i in range(num_epochs):
        counter += 1
        model, optimizer, losses = train(train_loader, model, optimizer, loss_fcn, use_prior, weight)
        # if epoch_i >= 20:
        #     swa_model.update_parameters(model)
        
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
    
    # model = swa_model
    return model, train_losses, valid_losses
    
if __name__ == "__main__":
    initial_rate = 1e-3
    wd = 1e-2
    N_EPOCHS = 100
    patience = 10

    dataset = sys.argv[1] #'both'
    BATCH_SIZE = int(sys.argv[2]) #32
    celltype = sys.argv[3] #'cd8'
    poolsize = int(sys.argv[4]) #2
    dropout = float(sys.argv[5]) #0.2
    use_prior = int(sys.argv[6])
    try:
        weight = float(sys.argv[7])  # fourier loss weighting
    except:
        weight = 1.0
    
    gc = ''
    ident = '_vi_150bp_aug'
    modelname = 'ad'

    basedir = f'/data/leslie/shared/ASA/mouseASA/{celltype}/cast'
    if use_prior:
        #fourier param
        freq_limit = 50
        limit_softness = 0.2
        att_prior_grad_smooth_sigma = 3
    else:
        print('no prior')
    print(modelname)

    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic=True
    if modelname=='ad':
        model = pairScan(poolsize, dropout, fc_train=False)    # change to True for fc training
    model.to(DEVICE)
    print(sum([p.numel() for p in model.parameters()]))

    loss_fcn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_rate, weight_decay=wd)

    x_train, x_valid, x_test, y_train, y_valid, y_test = load_data(celltype, dataset, gc+ident, frac=0.3)

    ## define the data loaders
    train_dataset = Dataset(x_train, y_train) #, get_confweights(dataset='train'))
    val_dataset = Dataset(x_valid, y_valid) #, get_confweights(dataset='val'))
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
    
    name = 'test_both_1'
    # SAVEPATH =  f'{basedir}/ckpt_models/{modelname}_{dataset}_{use_prior}_{BATCH_SIZE}_{gc}{ident}_fc.hdf5'
    SAVEPATH = f'{basedir}/ckpt_models/{name}.hdf5'
    # print(SAVEPATH)
    # model.load_state_dict(torch.load(SAVEPATH))
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, N_EPOCHS, optimizer, loss_fcn, SAVEPATH, patience, use_prior=bool(use_prior), weight=weight)
    model.load_state_dict(torch.load(SAVEPATH))
    # model.to('cpu')
    # torch.optim.swa_utils.update_bn(train_loader, model)
    # torch.save(model.state_dict(), SAVEPATH)
    
    # run testing with the trained model
    test_preds = test(test_loader, model)     # averaged over revcomps
    predsdir = f'{basedir}/preds/'
    print(test_preds.shape,'\n')
    if not os.path.exists(predsdir):
        os.makedirs(predsdir)
    # np.save(f'{predsdir}/{modelname}_{dataset}_{use_prior}_{BATCH_SIZE}_{gc}{ident}_fc.npy', test_preds)
    np.save(f'{predsdir}{name}.npy', test_preds)