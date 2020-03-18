import numpy as np
import argparse
import h5py
import ast
from function import *
from config import *
from cnn import *
from sklearn.utils import shuffle
from tqdm import tqdm
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
parser.add_argument('--n_filter',   type = int,   default = config['n_filter'],
                    help = 'number of filters')
parser.add_argument('--h_filter',   type = int,   default = config['h_filter'],
                    help = 'height of filters')
parser.add_argument('--w_filter',   type = int,   default = config['w_filter'],
                    help = 'width of filters')
parser.add_argument('--stride',   type = int,   default = config['stride'],
                    help = 'stride')
parser.add_argument('--padding',   type = int,   default = config['padding'],
                    help = 'padding')
parser.add_argument('--img_size',   type = int,   default = config['img_size'],
                    help = 'image size')
parser.add_argument('--debug',      type = ast.literal_eval, default = config['Debug'],     
                    dest = 'debug',
                    help = 'True or False flag, input should be either True or False.',
)
parser.add_argument('--isShuffle',      type = ast.literal_eval, default = config['isShuffle'],     
                    dest = 'isShuffle',
                    help = 'True or False flag, input should be either True or False.',
)


arg = parser.parse_args()
def one_hot_encode(y, num_class):
    m = y.shape[0]
    onehot = np.zeros((m, num_class), dtype="int32")
    for i in range(m):
        onehot[i][y[i]] = 1
    return onehot



def main():

    # generate seed for the same output as report
    np.random.seed(138)
    if arg.debug:
        arg.epochs = 1
    else:
        pass


    # load MNIST data
    MNIST_data = h5py.File('../MNISTdata.hdf5', 'r')
    x_train = np.float32(MNIST_data['x_train'][:])
    y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))
    x_test  = np.float32(MNIST_data['x_test'][:])
    y_test  = np.int32(np.array(MNIST_data['y_test'][:, 0]))
    MNIST_data.close()

    data_shape = (-1,1,28,28)
    x_train = x_train.reshape(data_shape)
    x_test = x_test.reshape(data_shape)
    # print(x_train.shape)
    # exit()

    # one-hot encoding
    examples = y_train.shape[0]
    # y_train = one_hot_encode(y_train, 10)
    # y_test  = one_hot_encode(y_test, 10)

    # number of training set
    # m_test = X.shape[0] - m
    # X_train, X_test = X[:m].T, X[m:].T
    # Y_train, Y_test = Y_new[:, :m], Y_new[:, m:]

    # initialization for CNN
    # cnn = CNN( mnist_dims,
    #            num_class = arg.num_class,
    #            n_filter = arg.n_filter,
    #            h_filter = arg.h_filter,
    #            w_filter = arg.w_filter,
    #            stride = arg.stride,
    #            padding = arg.padding)

    mnist_dims = (1, arg.img_size, arg.img_size)
    cnn = CNN(mnist_dims,
              num_class=arg.n_out,
              n_filter=arg.n_filter,
              h_filter=arg.h_filter,
              w_filter=arg.w_filter,
              stride=arg.stride,
              padding=arg.padding)

    # minibatches
    def batch(isShuffle):
        m = arg.training
        minibatches = []
        X,y = x_train.copy(),y_train.copy()
        if isShuffle:
            X, y = shuffle(X, y)
        for i in range(0,m,arg.batch_size):
            X_batch = X[i:i + arg.batch_size,:,:,:]
            y_batch = y[i:i + arg.batch_size,]
            minibatches.append((X_batch,y_batch))
        return minibatches

    def updata(grads):
        for param, grad in zip(cnn.params,reversed(grads)):
            for i in range(len(grad)):
                param[i] += -arg.lr * grad[i]

    batches = batch(arg.isShuffle)



    # initialization


    # training
    for n in tqdm(range(arg.epochs)):


        # training and update params
        for X_mini, y_mini in batches:
            loss, grads = cnn.train_step(X_mini, y_mini)
            for param, grad in zip(cnn.params,reversed(grads)):
                for i in range(len(grad)):
                    param[i] += -arg.lr * grad[i]

        train_acc = np.mean(y_train == cnn.predict(x_train))

        test_acc = np.mean(y_test == cnn.predict(x_test))
        print("Epoch {0}, Loss = {1}, Training Accuracy = {2}, Test Accuracy = {3}".format(
            n + 1, loss, train_acc, test_acc))


if __name__ == '__main__':
    main()
    


