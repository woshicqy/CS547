import numpy as np
from matplotlib import pyplot as plt
x1_resnet_cifar100 = np.load('renet_cifar100_train.npy')
x2_resnet_cifar100 = np.load('renet_cifar100_test.npy')
y_resnet_cifar100 = len(x1_resnet_cifar100)

x1_pretrained_cifar100 = np.load('pretrained_train.npy')
x2_pretrained_cifar100 = np.load('pretrained_test.npy')
y_pretrained_cifar100 = len(x1_pretrained_cifar100)

x1_resnet_tiny = np.load('tinyimage_train.npy')
x2_resnet_tiny = np.load('tinyimage_test.npy')
y_resnet_tiny = len(x1_resnet_tiny)

x1_sync_sgd = np.load('sync_sgd_train.npy')
x2_sync_sgd = np.load('sync_sgd_test.npy')
y_resnet_cifar100 = len(x1_resnet_cifar100)


Y1 = [x1_resnet_cifar100,x1_pretrained_cifar100,x1_resnet_tiny,x1_sync_sgd]
Y2 = [x2_resnet_cifar100,x2_pretrained_cifar100,x2_resnet_tiny,x2_sync_sgd]
X = [y_resnet_cifar100,y_pretrained_cifar100,y_resnet_tiny,y_resnet_cifar100]

name = ['resnet.png','pretrained.png','tiny.png','sync.png']
for i in range(4):
    plt.figure()
    plt.plot(X,X1[i],color = 'blue')
    plt.plot(X,Y2[i],color='orange')
    plt.legend(['Train','Test'])
    plt.title('Accuracy VS epoches')
    plt.xlabel('Numeber of epoches');
    plt.ylabel('Accuracy')
    plt.savefig(name[i])
plt.show()
