#!/usr/bin/env/ python
# ECBM E4040 Fall 2017 Assignment 2
# This Python script contains various functions for layer construction.

import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)

    :return:
    - out: output, of shape (N, M)
    - cache: x, w, b for back-propagation
    """
    #raise NotImplementedError
    #######################################################################
    #                                                                     #
    #                                                                     #
    #                         TODO: YOUR CODE HERE                        #
    #                                                                     #
    #                                                                     #
    #######################################################################
    out=np.dot(x,w)+b
    cache=(x,w,b)
    return(out, cache)


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
                    x: Input data, of shape (N, d_1, ... d_k)
                    w: Weights, of shape (D, M)

    :return: a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    #raise NotImplementedError
    #######################################################################
    #                                                                     #
    #                                                                     #
    #                         TODO: YOUR CODE HERE                        #
    #                                                                     #
    #                                                                     #
    #######################################################################
    x,w,b=cache
    
    dx=np.dot(dout,(w.transpose()))
    D=w.shape[0]
    N=x.shape[0]
    dw=np.dot(x.transpose(),dout)
    db=np.dot(dout.transpose(),np.ones(N))
    return dx,dw,db
def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    :param x: Inputs, of any shape
    :return: A tuple of:
    - out: Output, of the same shape as x
    - cache: x for back-propagation
    """
    #raise NotImplementedError
    #######################################################################
    #                                                                     #
    #                                                                     #
    #                         TODO: YOUR CODE HERE                        #
    #                                                                     #
    #                                                                     #
    #######################################################################
    out=np.maximum(0,x)
    cache=x
    return out,cache

def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    :param dout: Upstream derivatives, of any shape
    :param cache: Input x, of same shape as dout

    :return: dx - Gradient with respect to x
    """
    #raise NotImplementedError
    #######################################################################
    #                                                                     #
    #                                                                     #
    #                         TODO: YOUR CODE HERE                        #
    #                                                                     #
    #                                                                     #
    #######################################################################
    x=cache
    dout[x<=0]=0
    dx=dout
    return dx
def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    y_prediction = argmax(softmax(x))

    :param x: (float) a tensor of shape (N, #classes)
    :param y: (int) ground truth label, a array of length N

    :return: loss - the loss function
             dx - the gradient wrt x
    """
    #raise NotImplementedError
    #######################################################################
    #                                                                     #
    #                                                                     #
    #                         TODO: YOUR CODE HERE                        #
    #                                                                     #
    #                                                                     #
    #######################################################################
    N=x.shape[0]

    
    x-=np.max(x,axis=1,keepdims=True)
    temp=np.exp(x)
    dr_vec=np.sum(temp,axis=1,keepdims=True)

    nr=(x[np.arange(N),y]).reshape([N,1])
    loss=np.sum(-(nr)+np.log(dr_vec))
    
    loss=(loss/N)
    temp/=dr_vec
    temp[np.arange(N),y] -= 1
    
    dx = temp/N
    
    return loss, dx

def conv2d_forward(x, w, b, pad, stride):
    """
    A Numpy implementation of 2-D image convolution.
    By 'convolution', simple element-wise multiplication and summation will suffice.
    The border mode is 'valid' - Your convolution only happens when your input and your filter fully overlap.
    Another thing to remember is that in TensorFlow, 'padding' means border mode (VALID or SAME). For this practice,
    'pad' means the number rows/columns of zeroes to to concatenate before/after the edge of input.

    Inputs:
    :param x: Input data. Should have size (batch, height, width, channels).
    :param w: Filter. Should have size (num_of_filters, filter_height, filter_width, channels).
    :param b: Bias term. Should have size (num_of_filters, ).
    :param pad: Integer. The number of zeroes to pad along the height and width axis.
    :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

    :return: A 4-D array. Should have size (batch, new_height, new_width, num_of_filters).

    Note:
    To calculate the output shape of your convolution, you need the following equations:
    new_height = ((height - filter_height + 2 * pad) // stride) + 1
    new_width = ((width - filter_width + 2 * pad) // stride) + 1
    For reference, visit this website:
    https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/
    """
    #raise NotImplementedError
    

            
    #######################################################################
    #                                                                     #
    #                                                                     #
    #                         TODO: YOUR CODE HERE                        #
    #                                                                     #
    #                                                                     #
    #######################################################################
    ba,h,wd,c=x.shape
    f,fh,fw,c=w.shape
    n_h=((h-fh+2*pad)//stride)+1
    n_w=((wd-fw+2*pad)//stride)+1
    x_paded=np.pad(x,pad,'constant')
    temp_dim=x_paded.shape[3]
    #print(temp_dim)
    out=np.zeros((ba,n_h,n_w,f))
    for m in range(0,ba):
        for i in range(0,n_h):
            for j in range(0,n_w):
                for n in range(0,f):
                    h_t=i*stride
                    h_t2=i*stride+fh
                    w_t=j*stride
                    w_t2=j*stride+fw
                    temp=x_paded[pad+m,h_t:h_t2,w_t:w_t2,pad:temp_dim-pad]                    
                    out[m,i,j,n]=np.sum(temp*w[n,:,:,:])+b[n]
                    
    return out
            

def conv2d_backward(d_top, x, w, b, pad, stride):
    """
    (Optional, but if you solve it correctly, we give you +10 points for this assignment.)
    A lite Numpy implementation of 2-D image convolution back-propagation.

    Inputs:
    :param d_top: The derivatives of pre-activation values from the previous layer
                       with shape (batch, height_new, width_new, num_of_filters).
    :param x: Input data. Should have size (batch, height, width, channels).
    :param w: Filter. Should have size (num_of_filters, filter_height, filter_width, channels).
    :param b: Bias term. Should have size (num_of_filters, ).
    :param pad: Integer. The number of zeroes to pad along the height and width axis.
    :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

    :return: (d_w, d_b), i.e. the derivative with respect to w and b. For example, d_w means how a change of each value
     of weight w would affect the final loss function.

    Note:
    Normally we also need to compute d_x in order to pass the gradients down to lower layers, so this is merely a
    simplified version where we don't need to back-propagate.
    For reference, visit this website:
    http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
    """
    raise NotImplementedError
    #######################################################################
    #                                                                     #
    #                                                                     #
    #                         TODO: YOUR CODE HERE                        #
    #                                                                     #
    #                                                                     #
    #######################################################################
