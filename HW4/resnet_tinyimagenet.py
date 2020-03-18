import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import os

# import matplotlib.pyplot as plt
import sys
import numpy as np
import random

import torchvision
import torchvision.transforms as transforms

#### for resnet18 fine tune
from torch.utils import model_zoo

from model_tiny import ResNet
from test import test
from checkpoint import load_checkpoint
from tqdm import tqdm

import torch.optim as optim
from checkpoint import save_checkpoint
import torch.nn as nn

import argparse
import ast
from matplotlib import pyplot as plt
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

########### preprocess ###############
def create_val_folder(val_dir):
    path = os.path.join(val_dir,'images')
    filename = os.path.join(val_dir,'val_annotation.txt')
    fp = open(filename,'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path,folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(path,img)):
            os.rename(os.path.join(path,img),os.path.join(newpath,img))
    return
#### build data loader ###
train_dir = 'data/tiny-imagenet-200/train'
val_dir = 'data/tiny-imagenet-200/val'
if 'val_' in os.listdir(val_dir + 'images/')[0]:
    create_val_folder(val_dir)
    val_dir = val_dir + 'images/'
else:
    val_dir = val_dir + 'images/'





def data_loader_and_transformer(train_dir,test_dir,fine_tune=False):
    train_data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_data_tranform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Data loader.
    train_dataset = torchvision.datasets.ImageFolder(
        train_dir,
        transform=train_data_transform)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )

    test_dataset = torchvision.datasets.ImageFolder(
        test_dir
        transform=test_data_tranform)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=2
    )

    return train_data_loader, test_data_loader


def main(train_dir,test_dir):
    # Set seed.
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Load data.
    print("Performing data loader...")
    train_data_loader, test_data_loader = data_loader_and_transformer(
                                                train_dir,test_dir, 
                                                fine_tune=args.fine_tune)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    for curt_epoch in tqdm(range(start_epoch, args.epochs)):
        train_acc = []
        test_acc = []
        resnet.train()

        ### avoiding overflow ###
        if curt_epoch > 10:
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if state['step'] >= 1024:
                        state['step'] = 1000


        running_loss = 0
        total_correct = 0
        total_samples = 0


        if args.lr_schedule:
            scheduler = optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1)
            scheduler.step()

        
        for batch_index, (images, labels) in enumerate(train_data_loader):
        
            ### For debug ###
            if args.debug and total_samples >= 10001:
                return
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = resnet(images)

            
            loss = criterion(outputs, labels.long())
            loss.backward()

            optimizer.step()

            # Loss.
            running_loss += loss.item()
            curt_loss = running_loss / (batch_index + 1)

            # Accuracy.
            _, predict_label = torch.max(outputs, 1)
            total_samples += labels.shape[0]
            total_correct += predict_label.eq(labels.long()).float().sum().item()
            accuracy = total_correct / total_samples
            
            # print('Training [epoch: %d, batch: %d] loss: %.3f, accuracy: %.5f' %
            #         (curt_epoch + 1, batch_index + 1, curt_loss, accuracy))
            
            # Update best accuracy and save checkpoint
            if accuracy > best_acc:
                best_acc = accuracy
                save_checkpoint(resnet, curt_epoch, best_acc)
        train_acc.append(accuracy)
        
        # Loss.
        print('Training [epoch: %d] loss: %.3f, accuracy: %.5f' %
                (curt_epoch + 1, curt_loss, accuracy))

        test_accuracy = test(resnet,criterion,
                        test_data_loader,
                        device,debug=args.debug)
        test_acc.append(test_accuracy)
        train_acc = np.array(train_acc)
        test_acc = np.array(test_acc)
        np.save('tinyimage_train.npy',train_acc)
        np.save('tinyimage_test.npy',test_acc)
    ################################

    # Testing.
    print("--Start testing after training...")
    final_test_acc = test(resnet,criterion,test_data_loader,device,debug=args.debug)
    
    print("--All work completed!--")

if __name__=="__main__":
    main()