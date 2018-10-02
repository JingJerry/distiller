from hyperopt import hp
from hyperopt import fmin
from hyperopt import tpe
import numpy as np
import argparse
import time
import os
import sys
import torch
import torch.nn
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
try:
    import distiller
except ImportError:
    sys.path.append(module_path)
    import distiller
import apputils
from models import ALL_MODEL_NAMES, create_model
import pprint

pp = pprint.PrettyPrinter(indent=4, width=100)
def float_range(val_str):
    val = float(val_str)
    if val < 0 or val >= 1:
        raise argparse.ArgumentTypeError('Must be >= 0 and < 1 (received {0})'.format(val_str))
    return val
parser = argparse.ArgumentParser(description='Distiller image classification model compression')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=ALL_MODEL_NAMES,
                    help='model architecture: ' +
                    ' | '.join(ALL_MODEL_NAMES) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--validation-size', '--vs', type=float_range, default=0.1,
                    help='Portion of training dataset to set aside for validation')
parser.add_argument('--deterministic', '--det', action='store_true',
                    help='Ensure deterministic execution for re-producible results.')

args = parser.parse_args()
args.dataset = 'imagenet'
model = create_model(True, args.dataset, args.arch, 0)
train_loader, val_loader, test_loader, _ = apputils.load_data(
        args.dataset, os.path.expanduser(args.data), args.batch_size,
        args.workers, args.validation_size, args.deterministic)
seed = 666
def objective(space):
    #TODO
    accuracy = 0
    alpha = 0.2 #the importance of inference time
    inference_time = 0.0
    return (1-accuracy) + (alpha* inference_time) # F(Acc, Inf. time) = Err + a * Inf. Time
def validate():
    for i, (inputs, labels) in enumerate(val_loader):
        outputs = model(inputs.cuda())
        pred = outputs.max(1)
def get_space():
    keys = model.state_dict().keys()
    space = ({})
    for key in keys:
        if 'conv' in key and 'weight' in key:
            space[key] = hp.uniform(key, 0, 1)
    return space
def main():
    space = get_space()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=10,
                rstate=np.random.RandomState(seed))
    print(best)
if __name__ == '__main__':
    main()
