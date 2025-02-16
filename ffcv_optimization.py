import numpy as np
import torch
from copy import deepcopy
from nets import ToyNet, ResNet, BasicBlock, SimpleNet

from torch.cuda.amp import autocast

def train_epoch(loader,loss_func,model,optimizer):
    '''
    PERFORMS GRADIENT DESCENT AND RETURNS THE AVERAGE BATCH LOSS AND THE ACCURACY
    '''
    model.train()
    loss_batch = [] # a list of the loss each batch
    correct, total = 0, 0
    for input,labels in loader:

        optimizer.zero_grad()
        with autocast():
            output = model(input)
            loss = loss_func(output, labels)
        loss.backward()
        optimizer.step()
        loss_batch.append( loss.detach().cpu().item() )

        correct += output.argmax(1).eq(labels).sum().cpu().item()
        total += input.shape[0]
    
    return np.mean(loss_batch), 100*correct/total

def eval_epoch(loader,loss_func,model):
    '''
    CALCULATES THE CURRENT LOSS OF THE DATA AND THE ACCURACY
    '''
    model.eval()
    loss_batch = []
    correct, total = 0, 0
    with torch.no_grad():
        for input,labels in loader:
            with autocast():
                output = model(input)
                loss = loss_func(output, labels)
            loss_batch.append( loss.detach().cpu().item() )

            correct += output.argmax(1).eq(labels).sum().cpu().item()
            total += input.shape[0]
    return np.mean(loss_batch), 100*correct/total

def minimize_loss(train_loader,test_loader,loss_func,model,optimizer,n_epochs,log_every,start_epoch=0):
    train_loss, train_ACC = [], []
    test_loss, test_ACC = [], []
    
    for ep in range(start_epoch,n_epochs):
        tr_L, tr_ACC = train_epoch(loader=train_loader,loss_func=loss_func,model=model,optimizer=optimizer)
        train_loss.append(tr_L)
        train_ACC.append(tr_ACC)

        te_L, te_ACC = eval_epoch(loader=test_loader,loss_func=loss_func,model=model)
        test_loss.append(te_L)
        test_ACC.append(te_ACC)
    
        if ((ep + 1) % log_every == 0):
            print(f'epoch:{ep+1} - train loss:{tr_L} - test loss: {te_L} | train ACC={tr_ACC}% - test ACC={te_ACC}%')
    
    return train_loss, test_loss, train_ACC, test_ACC

def init_model(name:str):
    if name == 'resnet18':
        return ResNet(BasicBlock, [2, 2, 2, 2],high_res=False)
    if name == 'toy':
        return ToyNet()
    if name == 'simple':
        return SimpleNet()

    raise ValueError(f"Unknown model: {name}.")


def Shrink_and_Perturb(model,model_name:str,shrink=0.4,perturb=0.1,device='not_needed'):
    network_state_dict_old = deepcopy(model.state_dict())   # old dict
    model = init_model(model_name)                          # initialize model
    model.to(memory_format=torch.channels_last).cuda()
    network_state_dict = deepcopy(model.state_dict())       # dict of the init model
    # creating new dict with the formula: shrink * old + perturb * init
    network_state_dict_new = {
        key:( shrink * network_state_dict_old[key] + perturb * network_state_dict[key] ) for key in network_state_dict_old
        }
    return network_state_dict_new                           # returning new dict
    


def reset(
        train_loader,test_loader,loss_func,
        model,model_name,
        optimizer,
        epoch_grid:list,log_every:int,
        shrink=0.4,perturb=0.1
        ):
    
    train_loss, train_ACC = [], []
    test_loss, test_ACC = [], []
    # getting the learning rate:
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    momentum = optimizer.state_dict()['param_groups'][0]['momentum']
    wd = optimizer.state_dict()['param_groups'][0]['weight_decay']

    i=-1 # in case len(epoch_grid)=2, and then we don't enter the loop
    for i,start_epoch in enumerate(epoch_grid[:-2]):
        tr_L, te_L, tr_ACC, te_ACC = minimize_loss(
            train_loader,test_loader,loss_func,
            model,
            optimizer,
            n_epochs=epoch_grid[i+1],log_every=log_every,start_epoch=start_epoch
            )
        
        # update the lists
        train_loss += tr_L
        test_loss += te_L
        train_ACC += tr_ACC
        test_ACC += te_ACC

        model.load_state_dict( Shrink_and_Perturb(model,model_name,shrink,perturb) )
        # resetting the optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=wd)

    # the last steps in the epoch_grid w/o reset
    tr_L, te_L, tr_ACC, te_ACC = minimize_loss(
        train_loader,test_loader,loss_func,
        model,
        optimizer,
        n_epochs=epoch_grid[i+2],log_every=log_every,start_epoch=epoch_grid[i+1]
        )
    
    # update the lists
    train_loss += tr_L
    test_loss += te_L
    train_ACC += tr_ACC
    test_ACC += te_ACC

    return train_loss, test_loss, train_ACC, test_ACC

##########################################################################
###########################################################################
##########################################################################

def reinitialize_smallest_weights_frac(model, fraction: float=1e-4):
    """
    Reinitialize the smallest weights in a PyTorch model using uniform distribution.
    
    Args:
        model (nn.Module): PyTorch neural network model
        fraction (float): Fraction of weights to reinitialize (between 0 and 1)
    """
    # if not 0 <= fraction <= 1:
    #     raise ValueError("Fraction must be between 0 and 1")
    
    # Collect all weights from the model
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only consider weight parameters, not biases
            all_weights.append(param.data.flatten())
    
    # Concatenate all weights into a single tensor
    all_weights_tensor = torch.cat(all_weights)
    
    # Calculate the number of weights to reinitialize
    num_weights = len(all_weights_tensor)
    num_reinit = int(fraction * num_weights)
    
    # Find the threshold value for the smallest weights
    threshold = torch.sort(torch.abs(all_weights_tensor))[0][num_reinit]
    
    # Reinitialize the smallest weights for each parameter
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Create a mask for weights smaller than threshold
            mask = torch.abs(param.data) <= threshold
            
            # Calculate initialization bounds based on the parameter's fan_in
            if len(param.shape) > 1:
                fan_in = param.shape[1]
                bound = 1 / np.sqrt(fan_in)
            else:
                bound = 1 / np.sqrt(param.shape[0])
            
            # Generate new weights using uniform distribution
            new_weights = torch.zeros_like(param.data).uniform_(-bound, bound)
            
            # Update only the masked weights
            param.data[mask] = new_weights[mask]

def reset_small_weights(
        train_loader,test_loader,loss_func,
        model,model_name,
        optimizer,
        epoch_grid:list,log_every:int,
        fraction=1e-4
        ):
    
    train_loss, train_ACC = [], []
    test_loss, test_ACC = [], []
    # getting the learning rate:
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    momentum = optimizer.state_dict()['param_groups'][0]['momentum']
    wd = optimizer.state_dict()['param_groups'][0]['weight_decay']

    i=-1 # in case len(epoch_grid)=2, and then we don't enter the loop
    for i,start_epoch in enumerate(epoch_grid[:-2]):
        tr_L, te_L, tr_ACC, te_ACC = minimize_loss(
            train_loader,test_loader,loss_func,
            model,
            optimizer,
            n_epochs=epoch_grid[i+1],log_every=log_every,start_epoch=start_epoch
            )
        
        # update the lists
        train_loss += tr_L
        test_loss += te_L
        train_ACC += tr_ACC
        test_ACC += te_ACC

        reinitialize_smallest_weights_frac(model, fraction)
        # resetting the optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=wd)

    # the last steps in the epoch_grid w/o reset
    tr_L, te_L, tr_ACC, te_ACC = minimize_loss(
        train_loader,test_loader,loss_func,
        model,
        optimizer,
        n_epochs=epoch_grid[i+2],log_every=log_every,start_epoch=epoch_grid[i+1]
        )
    
    # update the lists
    train_loss += tr_L
    test_loss += te_L
    train_ACC += tr_ACC
    test_ACC += te_ACC

    return train_loss, test_loss, train_ACC, test_ACC

###################################################################
###################################################################

def zero_weights_above_threshold(model, threshold=0.6):
    for name, param in model.named_parameters():
        if 'bn' not in name:
            mask = torch.abs(param.data) > threshold
            param.data[mask] = param.data[mask] *0.

def reset_weights_above_threshold(
        train_loader,test_loader,loss_func,
        model,model_name,
        optimizer,
        epoch_grid:list,log_every:int,
        threshold=0.6
        ):
    
    train_loss, train_ACC = [], []
    test_loss, test_ACC = [], []
    # getting the learning rate:
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    momentum = optimizer.state_dict()['param_groups'][0]['momentum']
    wd = optimizer.state_dict()['param_groups'][0]['weight_decay']

    i=-1 # in case len(epoch_grid)=2, and then we don't enter the loop
    for i,start_epoch in enumerate(epoch_grid[:-2]):
        tr_L, te_L, tr_ACC, te_ACC = minimize_loss(
            train_loader,test_loader,loss_func,
            model,
            optimizer,
            n_epochs=epoch_grid[i+1],log_every=log_every,start_epoch=start_epoch
            )
        
        # update the lists
        train_loss += tr_L
        test_loss += te_L
        train_ACC += tr_ACC
        test_ACC += te_ACC

        zero_weights_above_threshold(model, threshold)
        # resetting the optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=wd)

    # the last steps in the epoch_grid w/o reset
    tr_L, te_L, tr_ACC, te_ACC = minimize_loss(
        train_loader,test_loader,loss_func,
        model,
        optimizer,
        n_epochs=epoch_grid[i+2],log_every=log_every,start_epoch=epoch_grid[i+1]
        )
    
    # update the lists
    train_loss += tr_L
    test_loss += te_L
    train_ACC += tr_ACC
    test_ACC += te_ACC

    return train_loss, test_loss, train_ACC, test_ACC


def Shrink_and_Perturb_last_layer(model,shrink=0.4,perturb=0.1,device='cuda'):
    network_state_dict_old = deepcopy(model.state_dict())   # old dict
    model = ResNet(BasicBlock, [2, 2, 2, 2],high_res=False,num_classes=10)    # initialize model
    model.to(device)
    network_state_dict = deepcopy(model.state_dict())       # dict of the init model
    # creating new dict with the formula: shrink * old + perturb * init
    L = len(network_state_dict_old)
    Shrink = np.ones(L)
    Shrink[-2:] = shrink*np.ones(2)
    Perturb = np.zeros(L)
    Perturb[-2:] = perturb*np.ones(2)
    network_state_dict_new = {
        key:( Shrink[i] * network_state_dict_old[key] + Perturb[i] * network_state_dict[key] ) for i,key in enumerate(network_state_dict_old)
        }
    return network_state_dict_new                           # returning new dict
    


def reset_last_layer(
        train_loader,test_loader,loss_func,
        model,
        optimizer,
        epoch_grid:list,log_every:int,
        shrink=0.4,perturb=0.1
        ):
    
    train_loss, train_ACC = [], []
    test_loss, test_ACC = [], []
    # getting the learning rate:
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    momentum = optimizer.state_dict()['param_groups'][0]['momentum']

    i=-1 # in case len(epoch_grid)=2, and then we don't enter the loop
    for i,start_epoch in enumerate(epoch_grid[:-2]):
        tr_L, te_L, tr_ACC, te_ACC = minimize_loss(
            train_loader,test_loader,loss_func,
            model,
            optimizer,
            n_epochs=epoch_grid[i+1],log_every=log_every,start_epoch=start_epoch
            )
        
        # update the lists
        train_loss += tr_L
        test_loss += te_L
        train_ACC += tr_ACC
        test_ACC += te_ACC

        model.load_state_dict( Shrink_and_Perturb_last_layer(model,shrink,perturb) )
        # resetting the optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # the last steps in the epoch_grid w/o reset
    tr_L, te_L, tr_ACC, te_ACC = minimize_loss(
        train_loader,test_loader,loss_func,
        model,
        optimizer,
        n_epochs=epoch_grid[i+2],log_every=log_every,start_epoch=epoch_grid[i+1]
        )
    
    # update the lists
    train_loss += tr_L
    test_loss += te_L
    train_ACC += tr_ACC
    test_ACC += te_ACC

    return train_loss, test_loss, train_ACC, test_ACC
