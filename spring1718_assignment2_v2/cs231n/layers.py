from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    origin_shape = x.shape
    x = x.reshape(x.shape[0], -1)
    out = x.dot(w) + b
    x = x.reshape(origin_shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    origin_shape = x.shape

    x = x.reshape(x.shape[0], -1) # input of shape (N, D)

    dw = x.T.dot(dout)

    db = dout.sum(axis = 0)

    dx = dout.dot(w.T)
    dx = dx.reshape(origin_shape)

    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(x, np.zeros_like(x))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout
    dx[x <= 0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, {}
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################

        # step 1
        mean = x.mean(axis=0)

        # step 2
        x_offset = x - mean 

        # step 3
        sq = x_offset ** 2

        # step 4
        var = sq.sum(axis=0) / N

        # step 5
        var_eps = var + eps

        # step 6
        sqrt = np.sqrt(var_eps)

        # step 7
        i_var = 1 / sqrt 

        # step 8
        _x = x_offset * i_var

        # step 9
        out = gamma * _x + beta

        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var

        # _x = (x - mean) / (var + eps)
        # out = gamma * _x + beta

        cache['gamma'] = gamma 
        cache['var_eps'] = var_eps
        cache['eps'] = eps 
        cache['x_offset'] = x_offset
        cache['sq'] = sq 
        cache['sqrt'] = sqrt 
        cache['i_var'] = i_var 
        cache['_x'] = _x 
        # cache['x'] = x

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        _x = (x - running_mean) / (np.sqrt(running_var) + eps)
        out = gamma * _x + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    N = dout.shape[0]

    # x = cache['x']
    x_offset = cache['x_offset']
    gamma = cache['gamma']
    sqrt = cache['sqrt']
    i_var = cache['i_var']
    var_eps = cache['var_eps']
    eps = cache['eps']
    sq = cache['sq']
    _x = cache['_x']

    # step 9
    dgamma = dout * _x
    dgamma = dgamma.sum(axis=0)
    dbeta = dout.sum(axis=0)

    # step 8
    d_x = gamma * dout 

    # step 7
    dx_offset = d_x * i_var

    # step 6
    di_var = (d_x * x_offset).sum(axis=0)
    dsqrt = -di_var / (sqrt**2)

    # step 5
    dvar = 0.5 * dsqrt / (np.sqrt(var_eps))

    # step 4
    dsq = dvar * np.ones_like(_x) / N

    # step 3
    dzero_centr= dsq * 2 * x_offset

    # step 2
    dx1 = dzero_centr + dx_offset
    dmean = -dx1.sum(axis=0)

    # step 1
    dx2 = dmean * np.ones_like(_x) / N

    # step 0
    dx = dx1 + dx2

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################

    N = dout.shape[0]

    x_offset = cache['x_offset']
    i_var    = cache['i_var']
    gamma    = cache['gamma']
    xhat     = cache['_x']

    # easy part
    dgamma = (dout * xhat).sum(axis=0)
    dbeta = dout.sum(axis=0)

    # single string dx
    dxhat = dout * gamma 
    dx = i_var * (N * dxhat - dxhat.sum(axis=0) - xhat * (dxhat * xhat).sum(axis=0)) / N


    # dvar/dxij
    # dvar = np.tile(-2*x_offset.sum(axis=0)/N, N).reshape(N, -1)
    # dvar = ((2*x_offset - dvar) / N).sum(axis=0)

    # # dsigma/dxij
    # dsigma = dvar/(2*np.sqrt(var_eps))

    # # dxhat
    # dxhat = ((1-1/N) - dsigma * x_offset) / var_eps

    # # dx
    # dx = dout * gamma * ((1-1/N) - ((2*x_offset - np.tile(-2*x_offset.sum(axis=0)/N, N).reshape(N, -1)) / N).sum(axis=0)/(2*np.sqrt(var_eps)) * x_offset) / var_eps


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, {}
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################

    N, D = x.shape

    # step 1
    mean = x.mean(axis=1).reshape(N, -1)

    # step 2
    x_offset = x - mean 

    # step 3
    sq = x_offset ** 2

    # step 4
    var = sq.sum(axis=1).reshape(N, -1) / D

    # step 5
    var_eps = var + eps

    # step 6
    sqrt = np.sqrt(var_eps)

    # step 7
    i_var = 1 / sqrt 

    # step 8
    _x = x_offset * i_var

    # step 9
    out = gamma * _x + beta

    cache['gamma'] = gamma 
    cache['var_eps'] = var_eps
    cache['eps'] = eps 
    cache['x_offset'] = x_offset
    cache['sq'] = sq 
    cache['sqrt'] = sqrt 
    cache['i_var'] = i_var 
    cache['_x'] = _x 

    # N, D = x.shape
    # mean = x.mean(axis=1).reshape(N, -1)
    # var = x.var(axis=1).reshape(N, -1)

    # xhat = (x - mean) / np.sqrt(var + eps)
    # out = gamma * xhat + beta

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################

    N, D = dout.shape

    i_var    = cache['i_var']
    gamma    = cache['gamma']
    xhat     = cache['_x']

    # easy part
    dgamma = (dout * xhat).sum(axis=0)
    dbeta = dout.sum(axis=0)

    # single string dx
    dxhat = dout * gamma 
    dx = i_var * (D * dxhat - dxhat.sum(axis=1).reshape(N, -1) - xhat * (dxhat * xhat).sum(axis=1).reshape(N, -1)) / D

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) <= p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = mask * dout
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    N, C, H, W = x.shape 
    F, C, HH, WW = w.shape 

    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)


    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (W + 2 * pad - WW) / stride

    if np.modf(H_out)[0] != 0 or np.modf(W_out)[0] != 0:
        print(H_out, W_out)
        raise ValueError('Invalid convolutional parameters. Output width and heigth can not be int.')
    
    H_out = int(H_out)
    W_out = int(W_out)

    out = np.zeros((N, F, H_out, W_out))
    x_with_pad = np.pad(array=x,  pad_width=[[0,0], [0,0], [pad, pad], [pad, pad]], mode='constant')

    for i in range(N):
        for fi in range(F):
            for hi in range(H_out):
                for wi in range(W_out):
                    out[i, fi, hi, wi] = \
                      (x_with_pad[i, : , hi*stride : hi*stride + HH, wi*stride : wi*stride + WW] * w[fi, ...]).sum() + b[fi]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache

    N, C, H, W = x.shape 
    F, C, HH, WW = w.shape 

    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)

    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)

    x_padded = np.pad(array=x,  pad_width=[[0,0], [0,0], [pad, pad], [pad, pad]], mode='constant')

    dx_padded = np.zeros_like(x_padded)
    db = np.zeros_like(b)
    dw = np.zeros_like(w) 

    # dx dw
    for i in range(N):
        for fi in range(F):
            for hi in range(H_out):
                for wi in range(W_out):
                    dx_padded[i, : , hi*stride : hi*stride + HH, wi*stride : wi*stride + WW] += \
                        w[fi, ...] * dout[i, fi, hi, wi]
                    dw[fi, ...] +=  \
                        x_padded[i, : , hi*stride : hi*stride + HH, wi*stride : wi*stride + WW] * dout[i, fi, hi, wi]

    dx = dx_padded[:, :, pad:-pad, pad:-pad]

    # db 
    db = dout.sum(axis=0).sum(axis=1).sum(axis=1)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape

    ph = pool_param.get('pool_height', 2)
    pw = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 1)

    H_out = 1 + (H - ph) / stride
    W_out = 1 + (W - pw) / stride

    if np.modf(H_out)[0] != 0 or np.modf(W_out)[0] != 0:
        raise ValueError('Invalid convolutional parameters. Output width and heigth can not be int.')

    H_out = int(H_out)
    W_out = int(W_out)

    out = np.zeros((N, C, H_out, W_out))
    for hi in range(H_out):
        for wi in range(W_out):
            out[:, :, hi, wi] = x[:, :, hi*stride : hi*stride + ph, wi*stride : wi*stride + pw].max(axis=2).max(axis=2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    pool_param['pool_height'] = ph 
    pool_param['pool_width'] = pw 
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache

    N, C, H, W = x.shape

    ph = pool_param.get('pool_height', 2)
    pw = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 1)

    H_out = 1 + (H - ph) / stride
    W_out = 1 + (W - pw) / stride

    if np.modf(H_out)[0] != 0 or np.modf(W_out)[0] != 0:
        raise ValueError('Invalid convolutional parameters. Output width and heigth can not be int.')

    H_out = int(H_out)
    W_out = int(W_out)

    dx = np.zeros_like(x) 

    for i in range(N):
        for ci in range(C):
            for hi in range(H_out):
                for wi in range(W_out):
                    # # indices of max along height (N, C, pool_width)
                    # ids_h = x[i, ci, hi*stride : hi*stride + ph, wi*stride : wi*stride + pw].argmax(axis=2) 
                    
                    # # indices of max along width (N, C, pool_height)
                    # ids_w = x[i, ci, hi*stride : hi*stride + ph, wi*stride : wi*stride + pw].argmax(axis=3) 

                    # very naive
                    idx_flatten = x[i, ci, hi*stride : hi*stride + ph, wi*stride : wi*stride + pw].argmax()
                    dx[i, ci, hi*stride + idx_flatten // ph, wi*stride + idx_flatten % ph] = \
                        1 * dout[i, ci, hi, wi]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    
    # We transpose to get channel as last dimension
    x_transposed = x.transpose((0, 2, 3, 1))
    # Then reshape for calculating mean and var along channels
    out, cache = batchnorm_forward(x_transposed.reshape((-1, x.shape[1])), gamma, beta, bn_param)
    # After that reshape back to desired shape
    out = out.reshape(x_transposed.shape).transpose((0, 3, 1, 2))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    
    # Take a look at forward pass for comments
    dout_transposed = dout.transpose((0, 2, 3, 1))
    dx, dgamma, dbeta = batchnorm_backward_alt(dout_transposed.reshape((-1, dout.shape[1])), cache)
    dx = dx.reshape(dout_transposed.shape).transpose((0, 3, 1, 2))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical 
    to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    
    # N, C, H, W = x.shape 
    # x_t = x.transpose((0, 2, 3, 1)) # now shape is (N, H, W, C)
    # gamma = gamma.flatten()
    # beta = beta.flatten()
    
    # caches = []
    # out = np.zeros_like(x_t)
    # for i in range(0, C, G):
    #     for_ln = x_t[:, :, :, i : i + G]

    #     out_part, cache = layernorm_forward(for_ln.reshape(-1, G), gamma[i : i + G], beta[i : i + G], gn_param)
        
    #     out[:, :, :, i : i + G] = out_part.reshape(for_ln.shape)
    #     caches.append(cache)
    
    # out = out.transpose((0, 3, 1, 2))

    N, C, H, W = x.shape
    x_t = x.transpose((0,2,3,1))
    for_ln = np.split(x_t, C // 2, axis=3)
    stacked = np.stack(for_ln)

    # step 1
    # mean = x.mean(axis=1).reshape(N, -1)

    groups_mean = stacked.mean(axis=-1).mean(axis=-1).mean(axis=-1)

    # step 2
    # x_offset = x - mean 

    centered = stacked - groups_mean[:, :, np.newaxis, np.newaxis, np.newaxis]

    # step 3
    sq = centered ** 2

    # step 4
    var = sq.sum(axis=-1).sum(axis=-1).sum(axis=-1) / (G * W * H)

    # step 5
    var_eps = var + eps

    # step 6
    sqrt = np.sqrt(var_eps)

    # step 7
    i_var = 1 / sqrt 

    # step 8
    print(centered.shape)
    x_hat = centered * i_var[:, :, np.newaxis, np.newaxis, np.newaxis]
    # reshape
    x_hat = np.concatenate(np.split(x_hat, C // G, axis=0), axis=-1)[0]
    x_hat = x_hat.transpose((0, 3, 1, 2))
    print(x_hat.shape)

    # step 9
    out = gamma * x_hat + beta

    cache = (gamma, var_eps, eps, centered, sq, sqrt, i_var, x_hat)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    
    N, C, H, W = dout.shape 
    dou_t = dout.transpose((0, 2, 3, 1)) # now shape is (N, H, W, C)
    gamma = gamma.flatten()
    beta = beta.flatten()
    
    dx, caches = np.zeros_like(dout), []
    for i in range(0, C, G):
        for_ln = dou_t[:, :, :, i : i + G]
        print(gamma.shape)

        dx_part, dgamma, dbeta = layernorm_forward(for_ln.reshape(-1, G), gamma[i : i + G], beta[i : i + G], gn_param)
        
        dout_t[:, :, :, i : i + G] = out.reshape(for_ln.shape)
        caches.append(cache)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
