from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    '''
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.amax(scores)
    
        scores_exp = np.sum(np.exp(scores))
        correct_exp = np.exp(scores[y[i]])

        loss -= np.log(correct_exp / scores_exp)


        for j in range(num_classes):
            if j == y[i]:
                continue
            dW[:,j] += np.exp(scores[j])/scores_exp * X[i]
        dW[:,y[i]] -= (scores_exp - correct_exp) / scores_exp * X[i]

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W*W)
    dW += 2*reg*W
    '''
    
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    scores = X.dot(W)
    scores -= np.amax(scores, axis = 1, keepdims = True)
 
    for i in range(num_train):  

        softmax = np.exp(scores[i]) / np.sum(np.exp(scores[i]))
        loss += -np.log(softmax[y[i]])

        for j in range(num_classes):
            dW[:,j] += X[i] * softmax[j] # softmax는 한 줄 데이터로 j번째 하나 요소를 사용하는것
        dW[:,y[i]] -= X[i]                # 결국은 퍼즐처럼 모든 dW가 계산됨
 
    loss /= num_train
    dW /= num_train
 
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W 
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    '''
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    '''
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    '''
    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X.dot(W)
    scores -= np.max(scores)

    scores_exp = np.exp(scores)
    scores_expsum = np.sum(scores_exp,axis = 1)
    correct_exp = scores_exp[range(num_train),y]

    loss = correct_exp / scores_expsum
    loss = -np.sum(np.log(loss))/num_train + reg*np.sum(W*W)

    s = np.divide(scores_exp, scores_expsum.reshape(num_train,1))
    s[range(num_train),y] = -(scores_expsum - correct_exp) / scores_expsum
    dW = X.T.dot(s)
    dW /= num_train
    dW += 2*reg*W
    '''


    num_train = X.shape[0]
    scores = X.dot(W)
    scores -= np.amax(scores, axis = 1, keepdims = True)

    scores_exp_sum = np.sum(np.exp(scores), axis = 1, keepdims = True)
    softmax = np.exp(scores) / scores_exp_sum # 브로드캐스팅
    loss = np.sum(-np.log(softmax[np.arange(num_train), y]))

    softmax[np.arange(num_train), y] -= 1
    dW = X.T.dot(softmax)
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W 
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
