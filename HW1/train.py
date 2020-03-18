from sklearn.metrics import classification_report
import numpy as np
import argparse
import h5py
import ast
from function import *
from config import *

################ IMPORTANT ###################
'''
I default debug state as True if you can run the code successfully
just run the code:
python -u train.py --debug False

Qiyang Chen ^_^
'''


parser = argparse.ArgumentParser()
# hyperparameters setting
parser.add_argument('--lr',         type = float, default = config['lr'],
                    help = 'learning rate')
parser.add_argument('--epochs',     type = int,   default = config['epochs'],
                    help = 'number of epochs to train')
parser.add_argument('--n_x',        type = int,   default = config['input_size'], 
                    help = 'number of inputs')
parser.add_argument('--n_h',        type = int,   default = config['hidden_units'],
                    help = 'number of hidden units')
parser.add_argument('--n_out',      type = int,   default = config['output_size'],
                    help = 'size of hidden layer output')
parser.add_argument('--beta',       type = float, default = config['beta'],
                    help = 'parameter for momentum')
parser.add_argument('--batch_size', type = int,   default = config['batch_size'],
                    help = 'input batch size')

parser.add_argument('--training',   type = int,   default = config['training_size'],
                    help = 'train size')

parser.add_argument('--debug',      type = ast.literal_eval, default = config['Debug'],     
                    dest = 'debug',
                    help = 'True or False flag, input should be either True or False.')
parser.add_argument('--test',      type = ast.literal_eval, default = config['Debug'],     
                    dest = 'test',
                    help = 'True or False flag, input should be either True or False.'
)
arg = parser.parse_args()

def main():

    # generate seed for the same output as report
    np.random.seed(138)
    if arg.debug:
        arg.epochs = 1
    else:
        pass


    # load MNIST data
    MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
    x_train = np.float32(MNIST_data['x_train'][:])
    y_train = np.int32(np.array(MNIST_data['y_train'][:, 0])).reshape(-1, 1)
    x_test  = np.float32(MNIST_data['x_test'][:])
    y_test  = np.int32(np.array(MNIST_data['y_test'][:, 0])).reshape(-1, 1)
    MNIST_data.close()


    # stack together for next step
    X = np.vstack((x_train, x_test))
    y = np.vstack((y_train, y_test))


    # one-hot encoding
    examples = y.shape[0]
    y = y.reshape(1, examples)
    Y_new = np.eye(arg.n_out)[y.astype('int32')]
    Y_new = Y_new.T.reshape(arg.n_out, examples)


    # number of training set
    m = arg.training
    m_test = X.shape[0] - m
    X_train, X_test = X[:m].T, X[m:].T
    Y_train, Y_test = Y_new[:, :m], Y_new[:, m:]


    # shuffle training set
    shuffle_index = np.random.permutation(m)
    X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]

    # number of batches
    batches = m // arg.batch_size

    # initialization
    params_a = {'w1': np.random.randn(arg.n_h, arg.n_x) * np.sqrt(1. / arg.n_x),
              'b1': np.zeros((arg.n_h, 1)) * np.sqrt(1. / arg.n_x),
              'w2': np.random.randn(arg.n_out, arg.n_h) * np.sqrt(1. / arg.n_h),
              'b2': np.zeros((arg.n_out, 1)) * np.sqrt(1. / arg.n_h)}


    params_b = {'w1': np.random.randn(100, 784) * np.sqrt(1. / 784),
              'b1': np.zeros((100, 1)) * np.sqrt(1. / 784),
              'w2': np.random.randn(arg.n_out, 100) * np.sqrt(1. / 100),
              'b2': np.zeros((arg.n_out, 1)) * np.sqrt(1. / 100)}

    params_c = {'w1': np.random.randn(60, 784) * np.sqrt(1. / 784),
              'b1': np.zeros((60, 1)) * np.sqrt(1. / 784),
              'w2': np.random.randn(arg.n_out, 60) * np.sqrt(1. / 60),
              'b2': np.zeros((arg.n_out, 1)) * np.sqrt(1. / 60)}


    ## model A
    dw1_a = np.zeros(params_a['w1'].shape)
    db1_a = np.zeros(params_a['b1'].shape)
    dw2_a = np.zeros(params_a['w2'].shape)
    db2_a = np.zeros(params_a['b2'].shape)


    ## model B
    dw1_b = np.zeros(params_b['w1'].shape)
    db1_b = np.zeros(params_b['b1'].shape)
    dw2_b = np.zeros(params_b['w2'].shape)
    db2_b = np.zeros(params_b['b2'].shape)

    ## model C
    dw1_c = np.zeros(params_c['w1'].shape)
    db1_c = np.zeros(params_c['b1'].shape)
    dw2_c = np.zeros(params_c['w2'].shape)
    db2_c = np.zeros(params_c['b2'].shape)


    # training
    for i in range(arg.epochs):

        # shuffle training set
        permutation = np.random.permutation(X_train.shape[1])
        X_train_shuffled = X_train[:, permutation]
        Y_train_shuffled = Y_train[:, permutation]

        if   (i + 1 == int(3*arg.epochs/10)):
            arg.lr = arg.lr / 10
        elif (i + 1 == int(7*arg.epochs/10)):
            arg.lr = arg.lr / 10
        elif (i + 1 == int(9*arg.epochs/10)):
            arg.lr = arg.lr / 10

        for j in range(batches):

            # get mini-batch
            begin = j * arg.batch_size
            end = min(begin + arg.batch_size, X_train.shape[1] - 1)
            X = X_train_shuffled[:, begin:end]
            Y = Y_train_shuffled[:, begin:end]
            m_batch = end - begin

            # forward and backward
            # model a
            cache_a = feed_forward(X, params_a)
            # model b
            cache_b = feed_forward(X, params_b)

            # model c
            cache_c = feed_forward(X, params_c)

            # ensemble training
            # just using cache_a but it doesn't matter. I will update it soon.
            cache_a['l2'] = cache_a['l2']\
                            + cache_b['l2']\
                            + cache_c['l2']
            

            sum_out = np.sum(cache_a['l2'],axis = 1,keepdims = True)

            C = 10000
            cache_a['l2'] = cache_a['l2']/sum_out
            cache_b['l2'] = cache_a['l2']/sum_out
            cache_c['l2'] = cache_a['l2']/sum_out

            
            grads_a = back_propagate(X, Y, params_a, cache_a, m_batch)
            grads_b = back_propagate(X, Y, params_b, cache_b, m_batch)
            grads_c = back_propagate(X, Y, params_c, cache_c, m_batch)


            # with momentum (argional)
            dw1_a = (arg.beta * dw1_a + (1. - arg.beta) * grads_a['dw1'])
            db1_a = (arg.beta * db1_a + (1. - arg.beta) * grads_a['db1'])
            dw2_a = (arg.beta * dw2_a + (1. - arg.beta) * grads_a['dw2'])
            db2_a = (arg.beta * db2_a + (1. - arg.beta) * grads_a['db2'])


            dw1_b = (arg.beta * dw1_b + (1. - arg.beta) * grads_b['dw1'])
            db1_b = (arg.beta * db1_b + (1. - arg.beta) * grads_b['db1'])
            dw2_b = (arg.beta * dw2_b + (1. - arg.beta) * grads_b['dw2'])
            db2_b = (arg.beta * db2_b + (1. - arg.beta) * grads_b['db2'])

            dw1_c = (arg.beta * dw1_c + (1. - arg.beta) * grads_c['dw1'])
            db1_c = (arg.beta * db1_c + (1. - arg.beta) * grads_c['db1'])
            dw2_c = (arg.beta * dw2_c + (1. - arg.beta) * grads_c['dw2'])
            db2_c = (arg.beta * db2_c + (1. - arg.beta) * grads_c['db2'])
            # gradient descent
            params_a['w1'] = params_a['w1'] - arg.lr * dw1_a
            params_a['b1'] = params_a['b1'] - arg.lr * db1_a
            params_a['w2'] = params_a['w2'] - arg.lr * dw2_a
            params_a['b2'] = params_a['b2'] - arg.lr * db2_a


            params_b['w1'] = params_b['w1'] - arg.lr * dw1_b
            params_b['b1'] = params_b['b1'] - arg.lr * db1_b
            params_b['w2'] = params_b['w2'] - arg.lr * dw2_b
            params_b['b2'] = params_b['b2'] - arg.lr * db2_b


            params_c['w1'] = params_c['w1'] - arg.lr * dw1_c
            params_c['b1'] = params_c['b1'] - arg.lr * db1_c
            params_c['w2'] = params_c['w2'] - arg.lr * dw2_c
            params_c['b2'] = params_c['b2'] - arg.lr * db2_c

        # forward pass on training set
        
        cache_a = feed_forward(X_train, params_a)
        train_loss = compute_loss(Y_train, cache_a['l2_out'])

        # forward pass on test set 
        
        cache_a = feed_forward(X_test, params_a)
        test_loss = compute_loss(Y_test, cache_a['l2_out'])
        print('Epoch {}: training loss = {}, test loss = {}'.format(
            i + 1, train_loss, test_loss))


    # test accuracy for each three models
    labels = np.argmax(Y_test, axis=0)

    cache_a = feed_forward(X_test, params_a)
    cache_b = feed_forward(X_test, params_b)
    cache_c = feed_forward(X_test, params_c)

    re_a = np.argmax(cache_a['l2_out'], axis=0)
    acc_a = np.sum(labels == re_a)/len(labels)


    re_b = np.argmax(cache_b['l2_out'], axis=0)
    acc_b = np.sum(labels == re_b)/len(labels)


    re_c = np.argmax(cache_c['l2_out'], axis=0)
    acc_c = np.sum(labels == re_c)/len(labels)
    print('Test_Acc_model_a: ',acc_a)
    print('Test_Acc_model_b: ',acc_b)
    print('Test_Acc_model_c: ',acc_c)

    # Try to use weight to ensemble models. 
    # Note: Weights I set here is based on nothing. I set them free.
    # And I miss other 3 cases. 

    if acc_b > acc_a and acc_b > acc_c:
        weight_b = 0.6
        weight_a = 0.2
        weight_c = 0.2
    elif acc_a > acc_b and acc_a > acc_c:
        weight_c = 0.2
        weight_b = 0.2
        weight_a = 0.6
    else:
        weight_c = 0.6
        weight_b = 0.2
        weight_a = 0.2


    # model ensemble
    final_out = weight_b * cache_a['l2_out'] \
              + weight_b * cache_b['l2_out'] \
              + weight_c * cache_c['l2_out']
    predictions = np.argmax(final_out, axis=0)
    
    # final results 
    print('Test_Acc: ',np.sum(labels == predictions)/len(labels))


if __name__ == '__main__':
    if arg.test:
        print('successfully!')
        exit()
    main()
