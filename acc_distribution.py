from typing import Any, List, Optional
import os
import numpy as np
import pandas as pd
from torch import Tensor
import torch
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm
from ffcv_optimization import train_epoch, eval_epoch, init_model, Shrink_and_Perturb
from ffcv_optimization import reinitialize_smallest_weights_frac

class ACC_DISTRIBUTION:
    """
    RECORDS THE TEST ACC VS. EPOCH FOR SEVERAL SEEDS. Currently only work with ffcv
    """
    def __init__(self, 
                 trainloader, testloader,
                 lr:float,momentum:float=0.9,
                 reset_every:Optional[int]=None, 
                 model_name:str = 'resnet18', 
                 shrink=0.4,perturb=0.1,
                 accuracy_condition:Optional[int]=None, 
                 n_epochs:int=100,
                 device = 'cuda',
                 use_ffcv = False,
                 dump_to_file:Optional[str]=None,
                 perturbation_kind='_perform_shrink_and_perturb',
                 fraction = 0.3
                 ):
        self.loss = nn.CrossEntropyLoss()
        self.trainloader = trainloader
        self.testloader = testloader
        self.lr = lr
        self.momentum = momentum
        self.reset_every = reset_every
        self.model_name = model_name
        self.shrink = shrink
        self.perturb = perturb
        self.accuracy_condition = accuracy_condition
        self.n_epochs = n_epochs
        self.device = device
        self.use_ffcv = use_ffcv
        self.perturbation_kind = perturbation_kind
        self.fraction = fraction
        # if use_ffcv:
        #     from ffcv_optimization import train_epoch, eval_epoch, init_model, Shrink_and_Perturb
        # else:
        #     from optimization import train_epoch, eval_epoch, init_model, Shrink_and_Perturb
        self.train_epoch=train_epoch
        self.eval_epoch = eval_epoch
        self.init_model = init_model
        self.Shrink_and_Perturb = Shrink_and_Perturb

        self.dump_to_file = dump_to_file
        if dump_to_file is not None:
            for i in range(1000):
                if not os.path.exists(dump_to_file.split('.csv')[0]+'_seed_'+str(i)+'.csv'):
                    break
            self.start_sample_from = i
    
    def _perform_perturbation(self):
        return getattr(self,self.perturbation_kind)()

    def _perform_shrink_and_perturb(self):
        self.network.load_state_dict( self.Shrink_and_Perturb(self.network,self.model_name,self.shrink,self.perturb,self.device) )
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
    
    def _perform_reset_small_weights(self):
        reinitialize_smallest_weights_frac(self.network,self.fraction)
    

    def _ffcv_train(self):
        self.test_acc_list = []
        for epoch in range(self.n_epochs):
            train_loss, train_ACC = self.train_epoch(loader=self.trainloader,loss_func=self.loss,model=self.network,optimizer=self.optimizer)

            test_loss, test_ACC = self.eval_epoch(loader=self.testloader,loss_func=self.loss,model=self.network)
            
            self.test_acc_list.append(test_ACC)

            if self.reset_every is not None:
                if (epoch+1)%self.reset_every==0:
                    self._perform_perturbation()
    
    def sample(self, n_samples:int=1000):
        for i in tqdm(range(self.start_sample_from,n_samples)):
            torch.manual_seed(i)
            self.network = self.init_model(self.model_name).to(memory_format=torch.channels_last).cuda()
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
            self._ffcv_train()
            filename = self.dump_to_file.split('.csv')[0]+'_seed_'+str(i)+'.csv'
            df = pd.DataFrame({'test_ACC':self.test_acc_list})
            df.to_csv(filename,index=False)
