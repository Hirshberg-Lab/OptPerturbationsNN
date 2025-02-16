import numpy as np
import torch
from copy import deepcopy
from nets import ToyNet, ResNet, BasicBlock, SimpleNet

def train_epoch(loader,loss_func,model,optimizer,device='cuda'):
    '''
    PERFORMS GRADIENT DESCENT AND RETURNS THE AVERAGE BATCH LOSS AND THE ACCURACY
    '''
    model.train()
    loss_batch = [] # a list of the loss each batch
    correct = 0
    for input,labels in loader:
        input = input.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        output = model(input)
        
        # loss
        loss = loss_func(output, labels)
        loss.backward()
        optimizer.step()
        loss_batch.append( loss.detach().cpu().item() )

        _, predicted = torch.max(output.data, 1)
        correct += (predicted == labels).sum().item()
    
    return np.mean(loss_batch), 100*correct/len(loader.dataset)

def eval_epoch(loader,loss_func,model,device='cuda'):
    '''
    CALCULATES THE CURRENT LOSS OF THE DATA AND THE ACCURACY
    '''
    model.eval()
    loss_batch = []
    correct = 0
    with torch.no_grad():
        for input,labels in loader:
            input = input.to(device)
            labels = labels.to(device)

            output = model(input)
            loss = loss_func(output, labels)
            loss_batch.append( loss.detach().cpu().item() )

            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum().item()
    return np.mean(loss_batch), 100*correct/len(loader.dataset)

def minimize_loss(train_loader,test_loader,loss_func,model,optimizer,n_epochs,log_every,start_epoch=0,device='cuda'):
    train_loss, train_ACC = [], []
    test_loss, test_ACC = [], []
    
    for ep in range(start_epoch,n_epochs):
        tr_L, tr_ACC = train_epoch(loader=train_loader,loss_func=loss_func,model=model,optimizer=optimizer,device=device)
        train_loss.append(tr_L)
        train_ACC.append(tr_ACC)

        te_L, te_ACC = eval_epoch(loader=test_loader,loss_func=loss_func,model=model,device=device)
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


def Shrink_and_Perturb(model,model_name:str,shrink=0.4,perturb=0.1,device='cuda'):
    network_state_dict_old = deepcopy(model.state_dict())   # old dict
    model = init_model(model_name)                          # initialize model
    model.to(device)
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
        shrink=0.4,perturb=0.1,
        device='cuda'
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
            n_epochs=epoch_grid[i+1],log_every=log_every,start_epoch=start_epoch,
            device=device
            )
        
        # update the lists
        train_loss += tr_L
        test_loss += te_L
        train_ACC += tr_ACC
        test_ACC += te_ACC

        model.load_state_dict( Shrink_and_Perturb(model,model_name,shrink,perturb,device) )
        # resetting the optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

    # the last steps in the epoch_grid w/o reset
    tr_L, te_L, tr_ACC, te_ACC = minimize_loss(
        train_loader,test_loader,loss_func,
        model,
        optimizer,
        n_epochs=epoch_grid[i+2],log_every=log_every,start_epoch=epoch_grid[i+1],
        device=device
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
        shrink=0.4,perturb=0.1,
        device='cuda'
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
            n_epochs=epoch_grid[i+1],log_every=log_every,start_epoch=start_epoch,
            device=device
            )
        
        # update the lists
        train_loss += tr_L
        test_loss += te_L
        train_ACC += tr_ACC
        test_ACC += te_ACC

        model.load_state_dict( Shrink_and_Perturb_last_layer(model,shrink,perturb,device) )
        # resetting the optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # the last steps in the epoch_grid w/o reset
    tr_L, te_L, tr_ACC, te_ACC = minimize_loss(
        train_loader,test_loader,loss_func,
        model,
        optimizer,
        n_epochs=epoch_grid[i+2],log_every=log_every,start_epoch=epoch_grid[i+1],
        device=device
        )
    
    # update the lists
    train_loss += tr_L
    test_loss += te_L
    train_ACC += tr_ACC
    test_ACC += te_ACC

    return train_loss, test_loss, train_ACC, test_ACC
