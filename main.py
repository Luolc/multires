"""Train CIFAR10 with PyTorch."""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from layerattn import *
from adabound import AdaBound


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--tag', default='', type=str, help='extra tag for checkpoint')
    parser.add_argument('--model', default='multi_resnet', type=str, help='model',
                        choices=['multi_resnet', 'resnet'])
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer',
                        choices=['sgd', 'adagrad', 'adam', 'amsgrad', 'adabound', 'amsbound'])
    parser.add_argument('--train_bsz', default=128, type=int, help='training batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--final_lr', default=0.1, type=float,
                        help='final learning rate of AdaBound')
    parser.add_argument('--gamma', default=1e-3, type=float,
                        help='convergence speed term of AdaBound')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay for optimizers')

    parser.add_argument('--n_layers', default=32, type=int, help='the number of layers')
    parser.add_argument('--pre_act', action='store_true', help='whether to use pre-activation')
    parser.add_argument('--kdim', default=32, type=int, help='dimension of attention keys')
    parser.add_argument('--mem_strategy', default='all', type=str,
                        help='the stategy of restoring the memory for attention',
                        choices=['all', 'one', 'two', 'same_dim'])

    return parser


def build_dataset(args):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bsz, shuffle=True,
                                               num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return train_loader, test_loader


def get_ckpt_name(tag='', model='multi_resnet', optimizer='sgd', lr=0.1, final_lr=0.1, momentum=0.9,
                  beta1=0.9, beta2=0.999, gamma=1e-3, mem_strategy='all'):
    name = {
        'sgd': 'lr{}-momentum{}'.format(lr, momentum),
        'adagrad': 'lr{}'.format(lr),
        'adam': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
        'amsgrad': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
        'adabound': 'lr{}-betas{}-{}-final_lr{}-gamma{}'.format(lr, beta1, beta2, final_lr, gamma),
        'amsbound': 'lr{}-betas{}-{}-final_lr{}-gamma{}'.format(lr, beta1, beta2, final_lr, gamma),
    }[optimizer]
    name = '{}-{}-{}-mem_{}'.format(model, optimizer, name, mem_strategy)

    if tag:
        name += '-' + tag

    return name


def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
    return torch.load(ckpt_name)


def build_model(args, device, ckpt=None):
    print('==> Building model..')
    net = {
        'multi_resnet': multi_resnet(args.n_layers, args.kdim, args.mem_strategy, args.pre_act),
        'resnet': resnet32(),
    }[args.model]
    net = net.to(device)
    if device == 'cuda':
        net = nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt['net'])

    return net


def create_optimizer(args, model_params):
    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        return optim.Adagrad(model_params, args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    elif args.optim == 'amsgrad':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, amsgrad=True)
    elif args.optim == 'adabound':
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.weight_decay)
    else:
        assert args.optim == 'amsbound'
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.weight_decay, amsbound=True)


def train(net, epoch, device, data_loader, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('train acc %.3f --- error %06.3f' % (accuracy, 100 - accuracy))

    return accuracy


def evaluate(net, device, data_loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(' test acc %.3f --- error %06.3f' % (accuracy, 100 - accuracy))

    return accuracy


def main():
    parser = get_parser()
    args = parser.parse_args()

    train_loader, test_loader = build_dataset(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('==> GPU: ' + str(torch.cuda.is_available()))

    ckpt_name = get_ckpt_name(tag=args.tag, model=args.model, optimizer=args.optim, lr=args.lr,
                              final_lr=args.final_lr, momentum=args.momentum,
                              beta1=args.beta1, beta2=args.beta2, gamma=args.gamma,
                              mem_strategy=args.mem_strategy)
    if args.resume:
        ckpt = load_checkpoint(ckpt_name)
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']
    else:
        ckpt = None
        best_acc = 0
        start_epoch = -1

    net = build_model(args, device, ckpt=ckpt)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, net.parameters())
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75, 100], gamma=0.25,
                                               last_epoch=start_epoch)

    train_accuracies = []
    test_accuracies = []

    for epoch in range(start_epoch + 1, 150):
        start_time = time.time()
        scheduler.step()
        train_acc = train(net, epoch, device, train_loader, optimizer, criterion)
        test_acc = evaluate(net, device, test_loader, criterion)

        # Save checkpoint.
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, os.path.join('checkpoint', ckpt_name))
            best_acc = test_acc

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        if not os.path.isdir('curve'):
            os.mkdir('curve')
        torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
                   os.path.join('curve', ckpt_name))

        elapsed_time = int(time.time() - start_time)
        print('[time] {:02d}:{:02d}'.format(elapsed_time // 60, elapsed_time % 60))


if __name__ == '__main__':
    main()
