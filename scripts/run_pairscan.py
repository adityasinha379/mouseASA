import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from model import pairScan
import os
from utils import place_tensor, fourier_att_prior_loss
from loaders import load_data_pairscan

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 0

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        'Initialization'
        self.x = x.swapaxes(2,3)  # (batch, 2, 4, 300)
        self.y = y
        
    def __len__(self):
        'Denotes the total number of samples'
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index].astype(np.float32), self.y[index].astype(np.float32)

def test(data_loader, model):
    model.eval()
    with torch.no_grad():
        preds = []
        for input_seqs,_ in data_loader:
            input_seqs = input_seqs.to(DEVICE)
            logit_pred_vals = model(input_seqs)
            preds.append(logit_pred_vals.detach().cpu().numpy())
    preds = np.concatenate(preds)
    preds = ((preds[:len(preds)//2]+preds[len(preds)//2:])/2).T.reshape(-1)        # average of revcomp preds
    # preds = preds[:len(preds)//2]
    return preds

def validate(data_loader, model, loss_fcn):
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():
        losses = []
        for input_seqs, output_vals in data_loader:
            input_seqs = input_seqs.to(DEVICE)
            output_vals = output_vals.to(DEVICE)

            logit_pred_vals = model(input_seqs)
            loss = loss_fcn(logit_pred_vals, output_vals)
            losses.append(loss.item())
    return model, losses

def train(data_loader, model, optimizer, loss_fcn, use_prior, weight=1.0):
    model.train()  # Switch to training mode
    losses = []
    cnt=0
    for input_seqs, output_vals in data_loader:
        cnt+=1
        optimizer.zero_grad()
        input_seqs = input_seqs.to(DEVICE)
        output_vals = output_vals.to(DEVICE)
        
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
    min_epochs = 30
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

    for epoch_i in range(num_epochs):
        counter += 1
        model, optimizer, losses = train(train_loader, model, optimizer, loss_fcn, use_prior, weight)
        # scheduler.step()
        if use_prior:
            fourier_losses = [x[1] for x in losses]
            losses = [x[0] for x in losses]
            train_loss_mean = np.mean(losses)
            train_fa_loss_mean = np.mean(fourier_losses)
            train_losses.append([train_loss_mean, train_fa_loss_mean])
        else:
            train_loss_mean = np.mean(losses)
            train_losses.append(train_loss_mean)
        
        # Run validation step
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
        if counter >= patience and epoch_i>=min_epochs:
            print('Val loss did not improve for {} epochs, early stopping...'.format(patience))
            break
    
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
    ident = '_vi_150bp_tn5_aug'
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
        model = pairScan(poolsize, dropout)    # change to True for fc training
        BATCH_SIZE//=2                 # paired input
    model.to(DEVICE)
    print(sum([p.numel() for p in model.parameters()]))

    loss_fcn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_rate, weight_decay=wd)

    x_train, x_valid, x_test, y_train, y_valid, y_test = load_data_pairscan(celltype, dataset, gc+ident, frac=0.3)

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
    
    if not os.path.exists(f'{basedir}/ckpt_models/'):
        os.makedirs(f'{basedir}/ckpt_models/')
    SAVEPATH =  f'{basedir}/ckpt_models/{modelname}_{dataset}_{BATCH_SIZE*2}_{weight}{gc}{ident}.hdf5'       # x2 because of paired input
    print(SAVEPATH)
    # model, train_losses, val_losses = train_model(model, train_loader, val_loader, N_EPOCHS, optimizer, loss_fcn, SAVEPATH, patience, use_prior=bool(use_prior), weight=weight)
    model.load_state_dict(torch.load(SAVEPATH))
    
    # run testing with the trained model
    test_preds = test(test_loader, model)     # averaged over revcomps
    predsdir = f'{basedir}/preds/'
    print(test_preds.shape,'\n')
    if not os.path.exists(predsdir):
        os.makedirs(predsdir)
    np.save(f'{predsdir}/{modelname}_{dataset}_{BATCH_SIZE*2}_{weight}{gc}{ident}.npy', test_preds)       # x2 because of paired input