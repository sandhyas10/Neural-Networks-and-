#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell


class MyLSTMCell(RNNCell):
    """
    Your own basic LSTMCell implementation that is compatible with TensorFlow. To solve the compatibility issue, this
    class inherits TensorFlow RNNCell class.

    For reference, you can look at the TensorFlow LSTMCell source code. To locate the TensorFlow installation path, do
    the following:

    1. In Python, type 'import tensorflow as tf', then 'print(tf.__file__)'

    2. According to the output, find tensorflow_install_path/python/ops/rnn_cell_impl.py

    So this is basically rewriting the TensorFlow LSTMCell, but with your own language.

    Also, you will find Colah's blog about LSTM to be very useful:
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """

    def __init__(self, num_units, num_proj, forget_bias=1.0, activation=None):
        """
        Initialize a class instance.

        In this function, you need to do the following:

        1. Store the input parameters and calculate other ones that you think necessary.

        2. Initialize some trainable variables which will be used during the calculation.

        :param num_units: The number of units in the LSTM cell.
        :param num_proj: The output dimensionality. For example, if you expect your output of the cell at each time step
                         to be a 10-element vector, then num_proj = 10.
        :param forget_bias: The bias term used in the forget gate. By default we set it to 1.0.
        :param activation: The activation used in the inner states. By default we use tanh.

        There are biases used in other gates, but since TensorFlow doesn't have them, we don't implement them either.
        """
        super(MyLSTMCell, self).__init__(_reuse=None)
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        self.num_units=num_units #size of c
        self.num_proj=num_proj #size of h
        self.forget_bias=forget_bias
        self.activation=activation
        self.w_f=tf.get_variable(name="forget_weights",shape=[self.num_proj+1,self.num_units],initializer=tf.glorot_uniform_initializer(seed=235))
        self.w_i=tf.get_variable(name="input_weights",shape=[self.num_proj+1,self.num_units],initializer=tf.glorot_uniform_initializer(seed=235))
        self.w_j=tf.get_variable(name="gate_weights",shape=[self.num_proj+1,self.num_units],initializer=tf.glorot_uniform_initializer(seed=235))
        self.w_o=tf.get_variable(name="output_weights",shape=[self.num_proj+1,self.num_units],initializer=tf.glorot_uniform_initializer(seed=235))
        self.w_h=tf.get_variable(name="proj_weights",shape=[self.num_units,self.num_proj],initializer=tf.glorot_uniform_initializer(seed=235))        
        #raise NotImplementedError('Please edit this function.')
    # The following 2 properties are required when defining a TensorFlow RNNCell.
    @property
    def state_size(self):
        """
        Overrides parent class method. Returns the state size of of the cell.

        state size = num_units + output_size

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        state_size=self.num_units+self.num_proj
        return(state_size)
        
        #raise NotImplementedError('Please edit this function.')

    @property
    def output_size(self):
        """
        Overrides parent class method. Returns the output size of the cell.

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        return(self.num_proj)
        #raise NotImplementedError('Please edit this function.')

    def call(self, inputs, state):
        """
        Run one time step of the cell. That is, given the current inputs and the state from the last time step,
        calculate the current state and cell output.

        You will notice that TensorFlow LSTMCell has a lot of other features. But we will not try them. Focus on the
        very basic LSTM functionality.

        Hint 1: If you try to figure out the tensor shapes, use print(a.get_shape()) to see the shape.

        Hint 2: In LSTM there exist both matrix multiplication and element-wise multiplication. Try not to mix them.

        :param inputs: The input at the current time step. The last dimension of it should be 1.
        :param state:  The state value of the cell from the last time step. The state size can be found from function state_size(self).
        :return: A tuple containing (output, new_state). For details check TensorFlow LSTMCell class.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        c_prev=tf.slice(state,[0, 0],[-1,self.num_units])
        h_prev=tf.slice(state,[0,self.num_units],[-1,self.num_proj])
        #c_prev,h_prev=state
        inputs_new=tf.concat([inputs,h_prev],1)
        temp=tf.matmul(inputs_new,self.w_f)
        ft=tf.sigmoid(temp+self.forget_bias)
        temp_2=tf.matmul(inputs_new,self.w_i)
        it=tf.sigmoid(temp_2)
        temp_3=tf.matmul(inputs_new,self.w_j)
        ct_temp=tf.tanh(temp_3)
        c_temp=c_prev*ft
        c_temp2=it*ct_temp
        c=c_temp+c_temp2
        temp_4=tf.matmul(inputs_new,self.w_o)
        ot=tf.sigmoid(temp_4)
        #print(ot.get_shape())
        #print((tf.tanh(c)*ot).get_shape())
        #print(self.w_h.get_shape())            
        h=tf.matmul(tf.tanh(c)*ot,self.w_h)
        state_updated=tf.concat([c,h],1)
        #print(h.get_shape())
        #print(state_updated.get_shape())
        return h,state_updated

    
    