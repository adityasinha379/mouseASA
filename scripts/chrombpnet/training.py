import torch
from datetime import datetime
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def test(data_loader, model):
    # Test only "current" model
    model.eval()
    with torch.no_grad():
        y_pred = []
        p_pred = []
        for x_b,_,_ in data_loader:
            x_b = x_b.to(DEVICE)
            temp = model(x_b)
            p_pred.append(temp[0].detach().cpu().numpy())
            y_pred.append(temp[1].detach().cpu().numpy())
    y_pred = np.concatenate(y_pred)
    p_pred = np.concatenate(p_pred)
    return y_pred, p_pred

def validate(data_loader, model, loss_fcn, modeltype):
    if modeltype=='full':
        model_bias = model[0]
        model = model[1]

    model.eval()  # Switch to evaluation mode
    with torch.no_grad():
        losses = []
        for x_b, y_b, p_b in data_loader:
            x_b = x_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            p_b = p_b.to(DEVICE)

            p_pred, y_pred = model(x_b)
            if modeltype=='full':
                p_pred_bias, y_pred_bias = model_bias(x_b)
                p_pred = p_pred + p_pred_bias                 # add profile outputs
                y_pred = torch.logsumexp(torch.stack((y_pred, y_pred_bias)), dim=0)
            
            loss = loss_fcn[0](y_b, y_pred) + loss_fcn[1](p_b, p_pred)
            losses.append(loss.item())
    return losses

def train(data_loader, model, optimizer, loss_fcn, modeltype):
    
    if modeltype=='full':
        model_bias = model[0]
        model = model[1]
    
    model.train()  # Switch to training mode
    losses = []

    for x_b, y_b, p_b in data_loader:
        x_b = x_b.to(DEVICE)
        y_b = y_b.to(DEVICE)
        p_b = p_b.to(DEVICE)
        optimizer.zero_grad()

        p_pred, y_pred = model(x_b)
        if modeltype=='full':
            p_pred_bias, y_pred_bias = model_bias(x_b)
            p_pred = p_pred + p_pred_bias                 # add profile outputs
            y_pred = torch.logsumexp(torch.stack((y_pred, y_pred_bias)), dim=0)
        
        nll_loss = loss_fcn[1](p_b, p_pred)
        loss = loss_fcn[0](y_b, y_pred) + nll_loss
        
        loss.backward()  # Compute gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10000)
        optimizer.step()  # Update weights through backprop
        losses.append([loss.item()-nll_loss.item(), nll_loss.item()])

    if modeltype=='full':
        return [model_bias, model], optimizer, losses
    else:
        return model, optimizer, losses


def train_model(model, train_loader, valid_loader, num_epochs, optimizer, loss_fcn, SAVEPATH, patience, modeltype):
    """
    Trains the model for the given number of epochs.
    """
    best_loss = 1e10
    train_losses = []
    valid_losses = []
    counter = 0
    min_epochs = 10

    for epoch_i in range(num_epochs):
        counter += 1
        model, optimizer, losses = train(train_loader, model, optimizer, loss_fcn, modeltype)
        
        nll_losses = [x[1] for x in losses]
        mse_losses = [x[0] for x in losses]
        train_loss_mean_y = np.mean(mse_losses)
        train_loss_mean_p = np.mean(nll_losses)
        train_losses.append([np.mean(mse_losses), np.mean(nll_losses)])
        
        losses = validate(valid_loader, model, loss_fcn, modeltype)
        valid_loss_mean = np.mean(losses)
        valid_losses.append(valid_loss_mean)
            
        if valid_loss_mean < best_loss:
            counter = 0
            print('++++ Val loss improved from {}, saving+++++'.format(best_loss))
            best_loss = valid_loss_mean
            try:
                asd
                # if modeltype=='full':
                #     torch.save(model[1].state_dict(), SAVEPATH)
                # else:
                #     torch.save(model.state_dict(), SAVEPATH)
            except:
                print('Failed to save.')
        
        print(f'{datetime.now().time().replace(microsecond=0)} --- '
            f'Epoch: {epoch_i}\t'
            f'Train loss: {(train_loss_mean_y+train_loss_mean_p):.3f} ({train_loss_mean_y:.3f}+{train_loss_mean_p:.3f})\t'
            f'Valid loss: {valid_loss_mean:.3f}\t'
        )
        
        if counter >= patience and epoch_i>min_epochs:
            print('Val loss did not improve for {} epochs, early stopping...'.format(patience))
            break

    return model, train_losses, valid_losses