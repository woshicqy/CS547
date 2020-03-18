import numpy as np
import h5py
import time
import copy
from random import randint
import math
from tqdm import tqdm
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
hidden_layer = 0
hidden_input = 200
hidden_output = 10
model = {}
model['W1'] = np.random.randn(hidden_input,num_inputs) / np.sqrt(num_inputs)
model['W2'] = np.random.randn(hidden_output,hidden_input) / np.sqrt(hidden_input)
model_grads = copy.deepcopy(model)
########### initialize layers ##############
def initial_layers (hidden_layer):
    layers = []
    for layer in range(hidden_layer+1):
        layers.append('W%s'%(str(layer + 1)))
    return layers
########### step function ##################
def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    if np.isnan(np.sum(z)):
        exit()
    return ZZ
def forward(x,y, model,cur_layer):
    Z = np.dot(model[cur_layer], x)
    Z = Z / np.max(Z)
    p = softmax_function(Z)
    return p
def backward(x,y,p, model, model_grads,cur_layer):
    dZ = -1.0*p
    dZ[y] = dZ[y] + 1.0
    for i in range(num_outputs):
        model_grads[cur_layer][i,:] = dZ[i]*x
    return model_grads
import time
time1 = time.time()
############# hyperparameter ################
LR = .01
num_epochs = 60
layers = initial_layers(hidden_layer)
############# Train #########################
for epochs in (range(num_epochs)):
    #Learning rate schedule
    if (epochs + 1 == 45):
        LR = LR / 10
    elif (epochs + 1 == 30):
        LR = LR / 10
    elif (epochs + 1 == 15):
        LR = LR / 10
    total_correct = 0

    for n in tqdm(range( len(x_train))):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:]
        x_max = np.max(x)
        x_norm = x / x_max
        p1 = forward(x_norm, y, model,'W1')
        # print(x_norm)
        p1_norm = p1 / np.max(p1)
        p = forward(p1_norm, y, model,'W2')
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(p1_norm,y,p, model, model_grads,'W2')
        model['W2'] = model['W2'] + LR * model_grads['W2']

        model_grads = backward(x_norm,y,p1_norm, model, model_grads,'W1')
        model['W1'] = model['W1'] + LR * model_grads['W1']

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
    x_test_norm = x / np.max(x)
    p1 = forward(x_test_norm, y, model,'W1')
    p1_norm = p1 / np.max(p1)
    p = forward(p1_norm, y, model,'W2')
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print('Test_acc: ',f'{total_correct/np.float(len(x_test) )}' )
