import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import h5py
from model import alleleScan, place_tensor
import os


# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def revcomp(x,y):
    # augment data by adding reverse complements
    # output x: (2n, seqlen, 4)    y: (2n,)
    x = np.vstack( (x, np.flip(np.flip(x,axis=1),axis=2)) )    # reverse complement the sequences and stack them below already existing peak sequences
    y = np.concatenate((y,y))                    # simply duplicate the log-summed accessibility RPMs
    return x,y


def load_data(celltype, dataset, suff):
    basedir = '/data/leslie/shared/ASA/mouseASA/'
    datadir = basedir+'data/'+celltype+'/'
    with h5py.File(datadir+'data'+suff+'.h5','r') as f:
        if dataset=='both':
            xTr = np.vstack((f['x_train_b6'][()], f['x_train_cast'][()], f['x_train_unegs'][()]))
            yTr = np.concatenate((f['y_train_b6'][()], f['y_train_cast'][()], f['y_train_unegs'][()]))
            xVa = np.vstack((f['x_val_b6'][()], f['x_val_cast'][()], f['x_val_unegs'][()]))
            yVa = np.concatenate((f['y_val_b6'][()], f['y_val_cast'][()], f['y_val_unegs'][()]))
            xTe = np.vstack((f['x_test_b6'][()],f['x_test_cast'][()]))
            yTe = np.concatenate((f['y_test_b6'][()],f['y_test_cast'][()]))
        elif dataset=='b6':
            xTr = np.vstack((f['x_train_b6'][()], f['x_train_unegs'][()]))
            yTr = np.concatenate((f['y_train_b6'][()], f['y_train_unegs'][()]))
            xVa = np.vstack((f['x_val_b6'][()], f['x_val_unegs'][()]))
            yVa = np.concatenate((f['y_val_b6'][()], f['y_val_unegs'][()]))
            xTe = np.vstack((f['x_test_b6'][()],f['x_test_cast'][()]))
            yTe = np.concatenate((f['y_test_b6'][()],f['y_test_cast'][()]))
        xTr, yTr = revcomp(xTr, yTr)
        xVa, yVa = revcomp(xVa, yVa)
        xTe, yTe = revcomp(xTe, yTe)

    return xTr, xTe, xVa, yTr, yTe, yVa
    

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

def train(data_loader, model, optimizer, epoch_i, num_epochs, loss_fcn, use_prior=False):
    
    model.train()  # Switch to training mode
#     torch.set_grad_enabled(True)
    losses = []

    
    for input_seqs, output_vals in data_loader:
        optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        input_seqs = input_seqs.to(DEVICE)
        output_vals = output_vals.to(DEVICE)
        # Clear gradients from last batch if training

        if use_prior:
            input_seqs.requires_grad = True  # Set gradient required
            logit_pred_vals = model(input_seqs)
            # Compute the gradients of the output with respect to the input
            input_grads, = torch.autograd.grad(
                logit_pred_vals, input_seqs,
                grad_outputs=place_tensor(
                    torch.ones(logit_pred_vals.size())
                ),
                retain_graph=True, create_graph=True
                # We'll be operating on the gradient itself, so we need to
                # create the graph
            )
            input_grads = input_grads * input_seqs  # Gradient * input
            input_seqs.requires_grad = False  # Reset gradient required
            loss = loss_fcn(logit_pred_vals, output_vals) + \
                model.fourier_att_prior_loss(
                    output_vals, input_grads, freq_limit, limit_softness,
                    att_prior_grad_smooth_sigma
            )
        else:
            logit_pred_vals = model(input_seqs)
            loss = loss_fcn(logit_pred_vals, output_vals)

        loss.backward()  # Compute gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10000)
        optimizer.step()  # Update weights through backprop

        losses.append(loss.item())

    return model, optimizer, losses

def validate(data_loader, model, loss_fcn):
    model.eval()  # Switch to evaluation mode
#     torch.set_grad_enabled(False)
    losses = []

    for input_seqs, output_vals in data_loader:
        input_seqs = input_seqs.to(DEVICE)
        output_vals = output_vals.to(DEVICE)

        logit_pred_vals = model(input_seqs)
        loss = loss_fcn(logit_pred_vals, output_vals)

#         pred_vals.append(
#             logit_pred_vals.detach().cpu().numpy()
#         )
        losses.append(loss.item())
    return model, losses#, np.concatenate(true_vals), np.concatenate(pred_vals)

def test(data_loader, model):
    model.eval()
    with torch.no_grad():
        preds = []
        for input_seqs,_ in data_loader:
            input_seqs = input_seqs.to(DEVICE)
            preds.append(model(input_seqs).detach().cpu().numpy())
    preds = np.concatenate(preds)
    preds = (preds[:len(preds)//2]+preds[len(preds)//2:])/2        # average of revcomp preds
    preds = preds[:len(preds)//2]
    return preds

def train_model(
    model, train_loader, valid_loader, num_epochs, optimizer, loss_fcn, SAVEPATH, patience, use_scheduler, use_prior=False
):
    """
    Trains the model for the given number of epochs.
    """
    best_loss = 1e10
    train_losses = []
    valid_losses = []
    counter = 0
    for epoch_i in range(num_epochs):
        counter += 1
        model, optimizer, losses = train(train_loader, model, optimizer, epoch_i, num_epochs, loss_fcn, use_prior)
        train_loss_mean = np.mean(losses)
        train_losses.append(train_loss_mean)
        
        with torch.no_grad():
            _, valid_loss = validate(valid_loader, model, loss_fcn)
            valid_loss_mean = np.mean(valid_loss)
            valid_losses.append(valid_loss_mean)
            
        if valid_loss_mean < best_loss:
            counter = 0
            print('++++ Val loss improved from {}, saving+++++'.format(best_loss))
            best_loss = valid_loss_mean
            try:
                torch.save(model.state_dict(),
                           SAVEPATH)
            except:
                print('Failed to save.')
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
    
if __name__ == "__main__":
    print(DEVICE)
    initial_rate = 1e-3
    wd = 1e-3
    N_EPOCHS = 100
    RANDOM_SEED = 0
    patience = 10

    dataset = sys.argv[1] #'both'
    BATCH_SIZE = np.int(sys.argv[2]) #32
    celltype = sys.argv[3] #'cd8'
    poolsize = np.int(sys.argv[4]) #2
    dropout = np.float(sys.argv[5]) #0.2
    n_ensemble = np.int(sys.argv[6]) #3
    use_prior = np.int(sys.argv[7])
    gc = ''
    ident = '_vi'
    modelname = 'm3'

    basedir = '/data/leslie/shared/ASA/mouseASA/'
    if use_prior:
        print('fourier_prior')
    else:
        print('no prior')
    SAVEPATH = basedir+'ckpt_models/{}/{}_{}_{}_{}_{}{}{}.hdf5'.format(celltype, celltype, modelname, dataset, use_prior, BATCH_SIZE, gc, ident)

    #fourier param
    freq_limit = 150
    limit_softness = 0.2
    att_prior_grad_smooth_sigma = 3

    torch.manual_seed(RANDOM_SEED)

    model = alleleScan(poolsize, dropout)
    model.to(DEVICE)

    if modelname=='m3':
        loss_fcn = nn.MSELoss()
    elif modelname=='po':           # still experimenting with this...
        loss_fcn = nn.PoissonNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_rate, weight_decay=wd)

    x_train, x_test, x_valid, y_train, y_test, y_valid = load_data(celltype, dataset, gc+ident)

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

    # model, train_losses, val_losses = train_model(model, train_loader, val_loader, N_EPOCHS, optimizer, loss_fcn, SAVEPATH, patience, False, use_prior=np.bool(use_prior) )
    model.load_state_dict(torch.load(SAVEPATH))
    
    # run testing with the trained model
    test_preds = test(test_loader, model)     # averaged over revcomps
    predsdir = basedir+'data/'+celltype+'/preds/'
    print(test_preds.shape)
    if not os.path.exists(predsdir):
        os.makedirs(predsdir)
    np.save(predsdir+'preds_{}_{}_{}_{}{}{}_noavg.npy'.format(modelname, dataset, use_prior, BATCH_SIZE, gc, ident), test_preds)