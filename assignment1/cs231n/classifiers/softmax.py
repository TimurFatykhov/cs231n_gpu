import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  a = 3

  for i in range(num_train):
    scores = X[i].dot(W)
    max_score = scores.max()
    exps = np.exp(scores - max_score)
    sum_ex = np.sum(exps)
    for j in range(num_classes):
      if j == y[i]:
        dW[: , j] -= (X[i] * exps[y[i]] * (sum_ex - exps[j])) / (sum_ex * exps[y[i]])
        continue 
      dW[: , j] += (X[i] * exps[y[i]] * exps[j]) / (sum_ex * exps[y[i]])

    loss += -scores[y[i]] + max_score + np.log(sum_ex)

  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += reg * 2 * W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  # LOSS
  scores = X.dot(W)
  max_score = scores.max()
  exps = np.exp(scores - max_score)
  sum_ex = np.sum(exps, axis = 1)
  loss = np.sum( -scores[range(num_train) , y] + max_score + np.log(sum_ex) )

  loss /= num_train
  loss += reg * np.sum(W * W)

  # GRADIENT

  # 1-st method
  # for w_yi  : -exps[y[i]] * exps[j] + exps[y[i]] * sum_ex  / (sum_ex * exps[y[i]])
  # for w_j   : -exps[y[i]] * exps[j]                        / (sum_ex * exps[y[i]])
  #
  # coeffs = -exps * exps[range(num_train), y].reshape(-1, 1)
  # coeffs[range(num_train), y] += exps[range(num_train), y] * sum_ex
  # coeffs /= sum_ex.reshape(-1,1)
  # coeffs /= exps[range(num_train), y].reshape(-1, 1)

  # 2-nd method
  coeffs = exps / sum_ex.reshape(-1,1)
  coeffs[range(num_train), y] -= 1

  dW = X.T.dot(coeffs)

  dW /= num_train
  dW += reg * 2 * W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

