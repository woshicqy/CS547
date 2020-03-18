import numpy as np
from function import *
from config import *
def get_im2col_indices(x_shape, field_height=3, field_width=3, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height, dtype='int32'), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height, dtype='int32'), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width, dtype='int32'), int(out_height))
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C, dtype='int32'),
                  field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height=3, field_width=3, padding=1, stride=1):
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(
        x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(
        x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

class Conv():
    """Convolutional layer"""

    def __init__(self, X_dim, n_filter, h_filter, w_filter, stride, padding):

        # input dimension
        self.d_X, self.h_X, self.w_X = X_dim

        # filter dimension
        self.n_filter, self.h_filter, self.w_filter = n_filter, h_filter, w_filter

        # stride and padding
        self.stride, self.padding = stride, padding

        # weights and biases initialization 
        self.W = np.random.randn(
            n_filter, self.d_X, h_filter, w_filter) / np.sqrt(n_filter / 2.)
        self.b = np.zeros((self.n_filter, 1))
        self.params = [self.W, self.b]

        # spatial size of output
        self.h_out = int((self.h_X - h_filter + 2 * padding) / stride + 1)
        self.w_out = int((self.w_X - w_filter + 2 * padding) / stride + 1)

        # output dimension
        self.out_dim = (self.n_filter, self.h_out, self.w_out)

    def forward(self, X):

        self.n_X = X.shape[0]

        self.X_col = im2col_indices(X,
                                    self.h_filter,
                                    self.w_filter,
                                    stride=self.stride,
                                    padding=self.padding)

        W_row = self.W.reshape(self.n_filter, -1)

        out = W_row.dot(self.X_col) + self.b
        out = out.reshape(self.n_filter, self.h_out, self.w_out, self.n_X)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):

        dout_flat = dout.transpose(1, 2, 3, 0).reshape(self.n_filter, -1)

        dW = dout_flat.dot(self.X_col.T)
        dW = dW.reshape(self.W.shape)
        db = np.sum(dout, axis=(0, 2, 3)).reshape(self.n_filter, -1)

        W_flat = self.W.reshape(self.n_filter, -1)

        dX_col = W_flat.T.dot(dout_flat)
        shape = (self.n_X, self.d_X, self.h_X, self.w_X)
        dX = col2im_indices(dX_col,
                            shape,
                            self.h_filter,
                            self.w_filter,
                            self.padding,
                            self.stride)

        return dX, [dW, db]

class Flatten():

    def __init__(self):
        self.params = []

    def forward(self, X):

        self.X_shape = X.shape
        self.out_shape = (self.X_shape[0], -1)
        # reshape the output
        out = X.ravel().reshape(self.out_shape)
        self.out_shape = self.out_shape[1]
        return out

    def backward(self, dout):

        out = dout.reshape(self.X_shape)
        return out, ()


class FullyConnected():
    """Fully Connected layer"""

    def __init__(self, in_size, out_size):

        self.W = np.random.randn(in_size, out_size) / np.sqrt(in_size / 2.)
        self.b = np.zeros((1, out_size))
        self.params = [self.W, self.b]

    def forward(self, X):

        self.X = X
        out = self.X.dot(self.W) + self.b
        return out

    def backward(self, dout):

        dW = self.X.T.dot(dout)
        db = np.sum(dout, axis=0)
        dX = dout.dot(self.W.T)
        return dX, [dW, db]


class CNN():


    def __init__(self, mnist_dims, num_class, n_filter, h_filter,
                 w_filter, stride, padding, loss_func=SoftmaxLoss):


        # build layers
        self.layers = self.build_layers(mnist_dims,
                                        n_filter,
                                        h_filter,
                                        w_filter,
                                        stride,
                                        padding,
                                        num_class)

        # parameters
        self.params = []
        for layer in self.layers:
            self.params.append(layer.params)

        # loss function
        self.loss_func = loss_func

    def build_layers(self, mnist_dims, n_filter, h_filter,
                     w_filter, stride, padding, num_class):

        # Convolutional layer
        conv = Conv(mnist_dims, n_filter, h_filter, w_filter, stride, padding)

        # ReLU activation layer
        relu_conv = ReLU()

        # flatten parameters
        flat = Flatten()

        # Fully Connected layer
        fc = FullyConnected(np.prod(conv.out_dim), num_class)

        return [conv, relu_conv, flat, fc]

    def forward(self, X):
        for layer in self.layers:
            # perform forward pass in each layer
            X = layer.forward(X)
        return X

    def backward(self, dout):

        grads = []
        for layer in reversed(self.layers):
            # perform back propagation in each layer
            dout, grad = layer.backward(dout)
            grads.append(grad)
        return grads

    def train_step(self, X, y):

        # forward pass
        out = self.forward(X)

        # compute loss
        loss, dout = self.loss_func(out, y)

        # back propagation
        grads = self.backward(dout)

        return loss, grads

    def predict(self, X):

        X = self.forward(X)
        return np.argmax(softmax(X), axis=1)




