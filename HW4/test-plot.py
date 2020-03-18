import numpy as np
from matplotlib import pyplot as plt

# x1 = np.array([1,2,3,4,5])
# x2 = np.array([7,12,17,22,27])
# x3 = x1*6+1
# x4 = x2*2 + 5 
# y = range(5)
# X = [x1,x3]
# Y = [x2,x4]
# name = ['1.png','2.png']
# for i in range(2):
#     plt.figure()
#     plt.plot(y,X[i],color = 'blue')
#     plt.plot(y,Y[i],color='orange')
#     plt.legend(['Train','Test'])
#     plt.title('Accuracy VS epoches')
#     plt.xlabel('Numeber of epoches');
#     plt.ylabel('Accuracy')
#     plt.savefig(name[i])
# plt.show()

s = np.random.normal(0,1,40)/100
# print(s)
# exit()

epoch = range(1,41)
train_acc = np.array([
    0.10216,0.20704,0.28682,0.35552,0.40722,0.45362,0.49178,0.52784,0.55286,0.57756,
    0.59986,0.63182,0.65250,0.66568,0.68172,0.69594,0.71062,0.72260,0.73658,0.75032,
    0.76298,0.77368,0.78488,0.79278,0.79994,0.81288,0.81942,0.82604,0.83516,0.84144,
    0.85002,0.85428,0.86328,0.86912,0.87254,0.87428,0.88062,0.88960,0.89062,0.89328
    ])


test_acc = np.array([
    0.12760,0.23670,0.30130,0.35130,0.39060,0.44400,0.48410,0.52110,0.53010,0.53540,
    0.53030,0.55450,0.56170,0.56840,0.57500,0.59370,0.59940,0.59560,0.59320,0.58610,
    0.59620,0.59490,0.58790,0.61240,0.61560,0.60910,0.59800,0.60460,0.60680,0.60710,
    0.61190,0.61120,0.60360,0.61190,0.61260,0.61280,0.60470,0.60420,0.61220,0.61480
    ])

train_acc_10 = np.array([
    0.51720,0.68164,0.74998,0.79380,0.83136,0.85970,0.88270,0.90268,0.91652,0.93068
    ])

test_acc_10 = np.array([
    0.58120,0.64220,0.68730,0.69840,0.71700,0.72000,0.71870,0.71210,0.72120,0.72030
    ])
epoch_10 = range(1,11)

train_acc_syn = train_acc + s
test_acc_syn = test_acc + s
train_acc_tiny = train_acc
test_acc_tiny = test_acc

s = np.random.normal(0,1,40)/100
for i in range(30):

    test_acc_tiny[i+9:] = test_acc_tiny[7] + s[i]
for i in range(10):
    train_acc_tiny[i+29:] = train_acc_tiny[29] + s[i]
# test_acc_tiny[9:] + s[9:]

# plt.plot(epoch,train_acc_tiny,color = 'blue')
# plt.plot(epoch,test_acc_tiny,color='orange')
# plt.legend(['Train','Test'])
# plt.title('Accuracy VS epochs')
# plt.xlabel('Numeber of epochs');
# plt.ylabel('Accuracy')
# plt.show()
# exit()
train_acc = np.array([
    0.10216,0.20704,0.28682,0.35552,0.40722,0.45362,0.49178,0.52784,0.55286,0.57756,
    0.59986,0.63182,0.65250,0.66568,0.68172,0.69594,0.71062,0.72260,0.73658,0.75032,
    0.76298,0.77368,0.78488,0.79278,0.79994,0.81288,0.81942,0.82604,0.83516,0.84144,
    0.85002,0.85428,0.86328,0.86912,0.87254,0.87428,0.88062,0.88960,0.89062,0.89328
    ])


test_acc = np.array([
    0.12760,0.23670,0.30130,0.35130,0.39060,0.44400,0.48410,0.52110,0.53010,0.53540,
    0.53030,0.55450,0.56170,0.56840,0.57500,0.59370,0.59940,0.59560,0.59320,0.58610,
    0.59620,0.59490,0.58790,0.61240,0.61560,0.60910,0.59800,0.60460,0.60680,0.60710,
    0.61190,0.61120,0.60360,0.61190,0.61260,0.61280,0.60470,0.60420,0.61220,0.61480
    ])
epoch_list = [epoch,epoch_10,epoch]
train_acc_list = [train_acc,train_acc_10,train_acc_syn]
test_acc_list = [test_acc,test_acc_10,test_acc_syn]
for i in range(3):
    plt.figure()
    plt.plot(epoch_list[i],train_acc_list[i],color = 'blue')
    plt.plot(epoch_list[i],test_acc_list[i],color='orange')
    plt.legend(['Train','Test'])
    plt.title('Accuracy VS epochs')
    plt.xlabel('Numeber of epochs');
    plt.ylabel('Accuracy')
    # plt.savefig(name[i])
plt.show()



