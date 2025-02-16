from typing import List
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from acc_distribution import ACC_DISTRIBUTION
from input_pars import args
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#################################
######### LOADING THE DATA ######
#################################

loaders = {}
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
print('using ffcv for faster training')

# Note that statistics are wrt to uin8 range, [0,255].
CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]

BATCH_SIZE = 125


for name in ['train', 'test']:
    label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(torch.device('cuda')), Squeeze()]
    image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

    image_pipeline.extend([
        ToTensor(),
        ToDevice(torch.device('cuda'), non_blocking=True),
        ToTorchImage(),
        Convert(torch.float16),
        torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # Create loaders
    ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL
    loaders[name] = Loader(f'./tmp/cifar_{name}.beton',
                            batch_size=BATCH_SIZE,
                            num_workers=8,
                            order=ordering,
                            drop_last=(name == 'train'),
                            pipelines={'image': image_pipeline,
                                    'label': label_pipeline})

#################################
#### INIT THE SAMPLING CLASS ####
#################################
if args.reset_rate is None:
    file_name = args.url+'test_acc_No_reset.csv'
else:
    file_name = args.url+'test_acc_reset_'+str(args.reset_rate)+'.csv'

acc_lists = ACC_DISTRIBUTION(
    trainloader=loaders['train'], testloader=loaders['test'],
    lr = args.lr ,momentum= args.momentum,
    reset_every=args.reset_rate, 
    model_name = args.model_name, 
    shrink=args.shrink,perturb=args.perturb,
    accuracy_condition=args.acc_condition, 
    n_epochs=args.n_epochs,
    device = device,
    use_ffcv = args.use_ffcv,
    dump_to_file = file_name,
    perturbation_kind=args.perturbation_kind,
    fraction = args.fraction
    )

#################################
##### STARTING THE SAMPLING #####
#################################

acc_lists.sample(args.n_samples)