import numpy as np
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
    """
    Implement im2col util function
    """
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
    """
    Implement col2im based on fancy indexing and np.add.at
    """
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
        """
        Initialize a Convolutional layer
        """
        # input dimension
        self.d_X, self.h_X, self.w_X = X_dim

        # filter dimension
        self.n_filter, self.h_filter, self.w_filter = n_filter, h_filter, w_filter

        # stride and padding
        self.stride, self.padding = stride, padding

        # weights and biases
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
        """
        Forward pass of conv layer
        """
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
        """
        Back propagation of conv layer
        """
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





