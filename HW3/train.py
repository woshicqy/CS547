import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import os.path
import argparse


from torch.autograd import Variable
from cnn import *
from config import *
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',   type=str,     default = config['dataroot'], 
                    help = 'path to dataset')
parser.add_argument('--ckptroot',   type=str,     default = config['ckptroot'], 
                    help = 'path to checkpoint')

# hyperparameters settings
parser.add_argument('--lr',         type = float, default = config['lr'], 
                    help = 'learning rate')
parser.add_argument('--wd',         type = float, default = config['wd'], 
                    help = 'weight decay')
parser.add_argument('--epochs',     type = int,   default = config['epochs'],
                    help = 'number of epochs to train')

parser.add_argument('--batch_size', type = int,   default = config['batch_size'], 
                    help = 'training set input batch size')
parser.add_argument('--input_size', type = int,   default = config['input_size'], 
                    help = 'size of input images')

# loading set 
parser.add_argument('--resume',     type = bool,  default = config['resume'],
                    help = 'whether training from ckpt')
parser.add_argument('--is_gpu',     type = bool,  default = config['is_gpu'],
                    help = 'whether training using GPU')

# parse the arguments
arg = parser.parse_args()


# transform on images

print("==> Data Augmentation ...")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Loading -_- -_- -_- -_- -_-

print("==> Preparing CIFAR10 dataset ...")

trainset = torchvision.datasets.CIFAR10(
    root=arg.dataroot, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=arg.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root=arg.dataroot, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=arg.batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Initialize model

print("==> Initialize CNN model ...")

start_epoch = 0

# resume training from the last time
if arg.resume:
    # Load checkpoint
    print('==> Resuming from checkpoint ...')
    assert os.path.isdir(
        '../checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(arg.ckptroot)
    net = checkpoint['net']
    start_epoch = checkpoint['epoch']
else:
    # start over
    print('==> Building new CNN model ...')
    net = CNN()
# To cuda
if arg.is_gpu:
    net = net.cuda()
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=arg.lr, weight_decay=arg.wd)


def calculate_accuracy(loader, is_gpu):
    correct = 0.
    total = 0.

    for data in loader:
        images, labels = data
        if is_gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    return 100 * correct / total


print("==> Start training ...")

for epoch in tqdm(range(start_epoch, arg.epochs + start_epoch)):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        if arg.is_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        inputs, labels = Variable(inputs), Variable(labels)
        # start to train
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if epoch > 16:
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if state['step'] >= 1024:
                        state['step'] = 1000
        optimizer.step()
        running_loss += loss.data[0]

    running_loss /= len(trainloader)

    # compute acc
    train_accuracy = calculate_accuracy(trainloader, arg.is_gpu)
    test_accuracy = calculate_accuracy(testloader, arg.is_gpu)

    print("Iteration: {0} | Loss: {1} | Training accuracy: {2}% | Test accuracy: {3}%".format(
        epoch+1, running_loss, train_accuracy, test_accuracy))

    # save model
    if epoch % 10 == 0:
        print('==> Saving model ...')
        state = {
            'net': net.module if arg.is_gpu else net,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, '../checkpoint/ckpt.t7')

print('==> Finished Training ...')
