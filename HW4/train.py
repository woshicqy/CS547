import torch
import torch.nn as nn
import torch.optim as optim

from test import test
from tqdm import tqdm
from checkpoint import save_checkpoint
import numpy as np


def train(
    net, 
    criterion, 
    optimizer,
    best_acc, 
    start_epoch, 
    epochs,
    train_data_loader,
    test_data_loader,
    device,
    lr_schedule=False,
    debug=False
    ):

    for curt_epoch in tqdm(range(start_epoch, epochs)):
        net.train()

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


        if lr_schedule:
            scheduler = optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1)
            scheduler.step()

        
        for batch_index, (images, labels) in enumerate(train_data_loader):
        
            ### For debug ###
            if debug and total_samples >= 10001:
                return
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)

            
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
                save_checkpoint(net, curt_epoch, best_acc)
        
        # Loss.
        print('Training [epoch: %d] loss: %.3f, accuracy: %.5f' %
                (curt_epoch + 1, curt_loss, accuracy))
        
        # Test every epoch.
        test(net, criterion, test_data_loader, device, debug=debug)
    
    print('Training finished')
