import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wst W, an array of same shape as W
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
    dr=0
    N=X.shape[0]
    C=W.shape[1]
    for i in range(N):
        p=X[i].dot(W)
        p_i=np.zeros_like(p)
        p-=np.max(p)
        dr=np.sum(np.exp(p))
        p_i=np.exp(p)/dr
        loss+=(-np.log(p_i[y[i]]))
        for j in range(C):
            if j!=y[i]:
                dW[:,j]+=p_i[j]*X[i]
            elif(j==y[i]):
                dW[:,j]+=(p_i[j]-1)*X[i]
    loss=(loss/N)+reg * np.sum(W * W)


    dW=dW/N+reg*W*2
   
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
    p_vec=np.dot(X,W)
    N=X.shape[0]

    temp=np.matrix(np.max(p_vec,axis=1))
    
    p_vec-=(temp.transpose())
    
    dr_vec=np.sum(np.exp(p_vec),axis=1)
    
    nr=(p_vec[np.arange(N),y])
    
    loss=np.sum(-np.log(np.exp(nr))+np.log(dr_vec))
    
    loss=(loss/N)+reg*np.sum(W*W)
    
    dr_vec=np.sum(np.exp(p_vec),axis=1,keepdims=True)


    p_class=np.exp(p_vec)/(dr_vec)


    temp=np.zeros(p_class.shape)
    temp[np.arange(N),y]=1
    dW=X.transpose().dot((p_class-temp))
    dW=dW/N+2*reg*np.sum(W)
  
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
