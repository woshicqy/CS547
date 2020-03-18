import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import matplotlib.pyplot as plt
import sys
import numpy as np
import random

import torchvision
import torchvision.transforms as transforms

#### for resnet18 fine tune
from torch.utils import model_zoo

from model import ResNet
from train import train
from test import test
from checkpoint import load_checkpoint

import argparse
import ast
####################################################
#                                                  # 
#  run coding as:                                  #
#  python main.py --debug False                    #
#                                                  #
####################################################

parser = argparse.ArgumentParser()

parser.add_argument("--fine_tune", default=False, type=ast.literal_eval,
                    dest =  "fine_tune",
                    help="fine-tune pretrained model")
parser.add_argument("--lr", default=0.001, type=float, 
                    help="learning rate")
parser.add_argument("--epochs", default=40, type=int, 
                    help="number of training epochs")
parser.add_argument("--lr_schedule", default=True, type=ast.literal_eval, 
                    dest =  "lr_schedule",
                    help="perform lr shceduling")
parser.add_argument("--load_checkpoint", default=False, type=ast.literal_eval, 
                    dest =  "load_checkpoint",                    
                    help="resume from checkpoint")
parser.add_argument("--show_sample_image", default=False, type=ast.literal_eval, 
                    dest =  "show_sample_image",
                    help="display data insights")
parser.add_argument("--debug", default=False, type=ast.literal_eval, 
                    dest =  "debug",
                    help="using debug mode")
parser.add_argument("--data_path", default="./data", type=str, 
                    help="path to store data")
args = parser.parse_args()

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def resnet18(pretrained=True) :
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir = './pretrained'))
    return model


def data_loader_and_transformer(root_path, fine_tune=False):
    if not fine_tune:
        train_data_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_data_tranform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    else:
        train_data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_data_tranform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    # Data loader.
    train_dataset = torchvision.datasets.CIFAR100(
        root=root_path,
        train=True,
        download=True,
        transform=train_data_transform
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root=root_path,
        train=False,
        download=True,
        transform=test_data_tranform
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=2
    )

    return train_data_loader, test_data_loader


def main():
    # Set seed.
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Load data.
    print("Performing data loader...")
    train_data_loader, test_data_loader = data_loader_and_transformer(
                                                args.data_path, 
                                                fine_tune=args.fine_tune)

    # Load sample image.
    if args.show_sample_image:
        print("Loading image sample from a batch...")
        data_iter = iter(train_data_loader)
        images, labels = data_iter.next()  # Retrieve a batch of data
        
        # Some insights of the data.
        # images type torch.FloatTensor, shape torch.Size([128, 3, 32, 32])
        print("images type {}, shape {}".format(images.type(), images.shape))
        # shape of a single image torch.Size([3, 32, 32])
        print("shape of a single image", images[0].shape)
        # labels type torch.LongTensor, shape torch.Size([128])
        print("labels type {}, shape {}".format(labels.type(), labels.shape))
        # label for the first 4 images tensor([12, 51, 91, 36])
        print("label for the first 4 images", labels[:4])
        
        # Get a sampled image.
        plt.imshow(images[0][0].numpy())
        plt.savefig("sample_image.png")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model.
    if args.fine_tune:
        print("*** Initializing pre-trained model...")
        resnet = resnet18()
        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, 100)
    else:
        print("*** Initializing model...")
        resnet = ResNet([2, 4, 4, 2])

    resnet = resnet.to(device)
    if device == 'cuda':
        resnet = torch.nn.DataParallel(resnet)
        cudnn.benchmark = True
    
    # Load checkpoint.
    start_epoch = 0
    best_acc = 0
    if args.load_checkpoint:
        print("*** Loading checkpoint...")
        start_epoch, best_acc = load_checkpoint(resnet)

    # Training.
    print('Start training on device {}...'.format(device))
    print('Hyperparameters: LR = {}, EPOCHS = {}, LR_SCHEDULE = {}'
          .format(args.lr, args.epochs, args.lr_schedule))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr=args.lr)
    train(
        resnet,
        criterion,
        optimizer,
        best_acc,
        start_epoch,
        args.epochs,
        train_data_loader,
        test_data_loader,
        device,
        lr_schedule=args.lr_schedule,
        debug=args.debug
    )

    # Testing.
    print("--Start testing...")
    test(
        resnet,
        criterion,
        test_data_loader,
        device,
        debug=args.debug
    )
    
    print("--All work completed!--")

if __name__=="__main__":
    main()