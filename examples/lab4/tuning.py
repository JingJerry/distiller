from hyperopt import hp
from hyperopt import fmin
from hyperopt import tpe
import math
import numpy as np
import argparse
import time
import os
import sys
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
try:
    import distiller
except ImportError:
    sys.path.append(module_path)
    import distiller
import apputils
from models import ALL_MODEL_NAMES, create_model

def float_range(val_str):
    val = float(val_str)
    if val < 0 or val >= 1:
        raise argparse.ArgumentTypeError('Must be >= 0 and < 1 (received {0})'.format(val_str))
    return val

parser = argparse.ArgumentParser(description='Distiller image classification model compression')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20_cifar',
                    choices=ALL_MODEL_NAMES,
                    help='model architecture: ' +
                    ' | '.join(ALL_MODEL_NAMES) +
                    ' (default: resnet20_cifar)')
parser.add_argument('--pretrained', action='store_true',
                    help='Using pretrained model')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--validation-size', '--vs', type=float_range, default=0.1,
                    help='Portion of training dataset to set aside for validation')
parser.add_argument('--deterministic', '--det', action='store_true',
                    help='Ensure deterministic execution for re-producible results.')
# Manual setting hyperparameters here
args = parser.parse_args()
args.dataset = 'cifar10' if 'cifar' in args.arch else 'imagenet'
args.epochs = 30
args.retrain_epoch = 25
args.max_iters = 30

model = create_model(args.pretrained, args.dataset, args.arch, 0)
train_loader, val_loader, test_loader, _ = apputils.load_data(
        args.dataset, os.path.expanduser(args.data), args.batch_size,
        args.workers, args.validation_size, args.deterministic)

count = 0
def objective(space):
    global count
    count += 1
    # Objective function: F(Acc, Lat) = (1 - Acc.) + (alpha * Lat.)
    accuracy = 0
    alpha = 0.2 # Super-parameter: the importance of inference time
    inference_time = 0.0
    # Training hyperparameter
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    #TODO: Pruning Scheduler Initialize
    """
    distiller/distiller/config.py
        # Element-wise sparsity
        sparsity_levels = {net_param: sparsity_level}
        pruner = distiller.pruning.SparsityLevelParameterPruner(name='sensitivity', levels=sparsity_levels)
        policy = distiller.PruningPolicy(pruner, pruner_args=None)
        scheduler = distiller.CompressionScheduler(model)
        scheduler.add_policy(policy, epochs=[0, 2, 4])
        # Structure sparsity and thinning
        # TODO
    """
    sparsity_levels = {}
    for key, value in space.items():
        sparsity_levels[key] = value
    pruner = distiller.pruning.SparsityLevelParameterPruner(name='sensitivity', levels=sparsity_levels)
    policy = distiller.PruningPolicy(pruner, pruner_args=None)
    lrpolicy = distiller.LRPolicy(torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1))
    compression_scheduler = distiller.CompressionScheduler(model)
    compression_scheduler.add_policy(policy, epochs=[args.retraining_epoch])
    compression_scheduler.add_policy(lrpolicy, starting_epoch=0, ending_epoch=args.epochs, frequency=1)
    #TODO: Pruning and Validate process
    """
    distiller/example/classifier_compression/compress_classifier.py
    For each epoch:
        compression_scheduler.on_epoch_begin(epoch)
        train()
        save_checkpoint()
        compression_scheduler.on_epoch_end(epoch)

    train():
        For each training step:
            compression_scheduler.on_minibatch_begin(epoch)
            output = model(input)
            loss = criterion(output, target)
            compression_scheduler.before_backward_pass(epoch)
            loss.backward()
            optimizer.step()
            compression_scheduler.on_minibatch_end(epoch)
    """
    for i in range(args.epochs):
        compression_scheduler.on_epoch_begin(i)
        train_accuracy = train(i,criterion, optimizer, compression_scheduler)
        val_accuracy = valid_accuracy(i) # Validate hyperparameter setting
        latency = valid_latency(i)
        compression_scheduler.on_epoch_end(i, optimizer)
        apputils.save_checkpoint(i, args.arch, model, optimizer, compression_scheduler, train_accuracy, False,
                                         'hyperopt', './')
    score = (1-(val_accuracy/100.)) + (alpha * latency * 1000) # convert to ms 
    print('{} trials: score: {}, train acc:{}, val acc:{}, latency:{}'.format(count, score, train_accuracy, val_accuracy, latency))
    return score
def train(epoch,criterion, optimizer, compression_scheduler):
    correct = 0
    total = 0
    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    for train_step, (inputs, targets) in enumerate(train_loader):
        compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().data.numpy()
        loss = criterion(outputs, targets)
        compression_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, loss,
                                                   optimizer=optimizer, return_loss_components=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)
    accuracy = 100. * correct / total    
    return accuracy
def valid_accuracy(epoch):
    model.eval() # Very Important 
    correct = 0
    total = 0
    with torch.no_grad():
        for test_step, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().data.numpy()
    accuracy = 100. * correct / total    
    return accuracy
    
def valid_latency(epoch):
    valid_times = 100
    latency = 0.0
    model.eval() # Very Important 
    with torch.no_grad():
        for i in range(valid_times):
            inputs = Variable(torch.randn(1,3,224,224)).cuda() if args.dataset == 'imagenet' else Variable(torch.randn(1,3,32,32)).cuda()
            start = time.time()
            model(inputs)
            latency += time.time() - start
    avg_latency = latency / valid_times
    return avg_latency
def get_space():
    keys = model.state_dict().keys()
    space = ({})
    for key in keys:
        if 'conv' in key and 'weight' in key:
            space[key] = hp.uniform(key, 0.45, 0.55)

    return space
def main():
    space = get_space()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=args.max_iters)
    print(best)
if __name__ == '__main__':
    main()
