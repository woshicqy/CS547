import numpy as np
import h5py
import time
import copy
from random import randint
#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()
#Implementation of stochastic gradient descent algorithm
#number of inputs
num_inputs = 28*28
#number of outputs
num_outputs = 10
model = {}
model['W1'] = np.random.randn(num_outputs,num_inputs) / np.sqrt(num_inputs)
model_grads = copy.deepcopy(model)
def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ
def forward(x,y, model):
    Z = np.dot(model['W1'], x)
    p = softmax_function(Z)
    return p
def backward(x,y,p, model, model_grads):
    dZ = -1.0*p
    dZ[y] = dZ[y] + 1.0
    for i in range(num_outputs):
        model_grads['W1'][i,:] = dZ[i]*x
    return model_grads
import time
time1 = time.time()
############# hyperparameter ################
LR = .01
num_epochs = 1
hidden_input = 10
hidden_layer = 1
hidden_output = 10
############# Train #########################
for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs + 1 == 15):
        LR = LR / 10
    elif (epochs + 1 == 10):
        LR = LR / 10
    elif (epochs + 1 == 5):
        LR = LR / 10
    total_correct = 0
    for n in range( len(x_train)):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:]
        p = forward(x, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x,y,p, model, model_grads)
        model['W1'] = model['W1'] + LR * model_grads['W1']
    if epochs % 5 == 0:
        print(f'Num_epochs: {epochs}')
        print(f'Train_acc: {total_correct / np.float(len(x_train) )}' )
time2 = time.time()
d_time = time2 - time1
print('Time: ',f'{d_time}')
######################################################
#test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    p = forward(x, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print('Test_acc: ',f'{total_correct/np.float(len(x_test) )}' )
