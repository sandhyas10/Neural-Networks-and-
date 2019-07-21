#!/usr/bin/env python
# ECBM E4040 Fall 2017 Assignment 2
# This script is intended for task 5 Kaggle competition. Use it however you want.
#sources: piazza and cnn_sample

import tensorflow as tf
import time
from ecbm4040.neuralnets.cnn_sample import *
import numpy as np
from ecbm4040.image_generator import ImageGenerator
def kaggle_LeNet(input_x, input_y,proba,
          img_len=96, channel_num=3, output_size=5,
          conv_featmap=[6, 16], fc_units=[84],
          conv_kernel_size=[5, 5], pooling_size=[2, 2],
          l2_norm=0.01, seed=235):
   
    assert len(conv_featmap) == len(conv_kernel_size) and len(conv_featmap) == len(pooling_size)

    print(input_x.shape)
    #norm layer
    norm_layer_0=norm_layer(input_x)
    print(norm_layer_0.output().shape)
    
    # conv layer

    
    conv_layer_0 = conv_layer(input_x=norm_layer_0.output(),
                              in_channel=channel_num,
                              out_channel=conv_featmap[0],
                              kernel_shape=conv_kernel_size[0],
                              rand_seed=seed,index=0)
    
    print(conv_layer_0.output().shape)
    #dropout layer
    dropout_layer_0=tf.nn.dropout(conv_layer_0.output(),proba)
    print(dropout_layer_0.shape)
    pooling_layer_0 = max_pooling_layer(input_x=dropout_layer_0,
                                        k_size=pooling_size[0],
                                        padding="VALID")
    
    print(pooling_layer_0.output().shape)
    
    norm_layer_1=norm_layer(pooling_layer_0.output())
    print(norm_layer_1.output().shape)
    conv_layer_1 = conv_layer(input_x=norm_layer_1.output(),
                              in_channel=conv_featmap[0],
                              out_channel=conv_featmap[1],
                              kernel_shape=conv_kernel_size[1],
                              rand_seed=seed,index=1)
    print(conv_layer_1.output().shape)
    #dropout_layer_1=tf.nn.dropout(conv_layer_1.output(),proba)
    #print(dropout_layer_1.shape)
    pooling_layer_1 = max_pooling_layer(input_x=conv_layer_1.output(),
                                        k_size=pooling_size[1],
                                        padding="VALID")
    print(pooling_layer_1.output().shape)
    # flatten
    pool_shape = pooling_layer_1.output().get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(pooling_layer_1.output(), shape=[-1, img_vector_length])

    # fc layer
    fc_layer_0 = fc_layer(input_x=flatten,
                          in_size=img_vector_length,
                          out_size=fc_units[0],
                          rand_seed=seed,
                          activation_function=tf.nn.relu,
                          index=0)

    fc_layer_1 = fc_layer(input_x=fc_layer_0.output(),
                          in_size=fc_units[0],
                          out_size=fc_units[1],
                          rand_seed=seed,
                          activation_function=None,
                          index=1)

    
    fc_layer_2 = fc_layer(input_x=fc_layer_1.output(),
                          in_size=fc_units[1],
                          out_size=output_size,
                          rand_seed=seed,
                          activation_function=None,
                          index=2)
    # saving the parameters for l2_norm loss
    conv_w = [conv_layer_1.weight,conv_layer_1.weight]
    fc_w = [fc_layer_0.weight, fc_layer_1.weight,fc_layer_2.weight]

    # loss
    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
        l2_loss += tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in conv_w])

        label = tf.one_hot(input_y, 5)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=fc_layer_2.output()),
            name='cross_entropy')
        loss = tf.add(cross_entropy_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('LeNet_loss', loss)

    return fc_layer_2.output(), loss

def kaggle_train_step(loss, learning_rate=1e-3):
    with tf.name_scope('kaggle_train_step'):
        step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return step

def kaggle_training(X_train, y_train, X_val, y_val, 
             conv_featmap=[6],
             fc_units=[84],
             conv_kernel_size=[5],
             pooling_size=[2],
             l2_norm=0.01,
             seed=235,
             learning_rate=1e-2,
             epoch=20,
             batch_size=245,
             verbose=False,
             pre_trained_model=None,pro=1.0):
    print("Building my LeNet. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))

    
    
    
    
    img1=ImageGenerator(X_train,y_train)
    img2=ImageGenerator(X_train,y_train)
    img3=ImageGenerator(X_train,y_train)
    
    img1.translate(16,16)
    img2.rotate(45.0)
    img3.flip("v")
              
    X_train=np.concatenate((X_train,img1.x,img2.x,img3.x),axis=0)
    y_train=np.concatenate((y_train,img1.y,img2.y,img3.y),axis=0)
    
    
    
    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, 96, 96, 3], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
        proba=tf.placeholder(tf.float32)
        
    
    output, loss = kaggle_LeNet(xs, ys,proba,
                         img_len=96,
                         channel_num=3,
                         output_size=5,
                         conv_featmap=conv_featmap,
                         fc_units=fc_units,
                         conv_kernel_size=conv_kernel_size,
                         pooling_size=pooling_size,
                         l2_norm=l2_norm,
                         seed=seed)
    
    

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    step = kaggle_train_step(loss,learning_rate)
    eve = evaluate(output, ys)

    iter_total = 0
    best_acc = 0
    cur_model_name = 'lenet_{}'.format(int(time.time()))

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                print("Load model Failed!")
                pass

        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))

            for itr in range(iters):
                iter_total += 1
                training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
                training_batch_y = y_train[itr * batch_size: (1 + itr) * batch_size]
                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x, ys: training_batch_y,proba:pro})
                if iter_total % 100 == 0:
                    # do validation
                    valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val, ys: y_val,proba:1.0})
                    valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                    if verbose:
                        print('{}/{} loss: {} validation accuracy : {}%'.format(
                            batch_size * (itr + 1),
                            X_train.shape[0],
                            cur_loss,
                            valid_acc))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy, we store the model paramters
                    if valid_acc > best_acc:
                        print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                        best_acc = valid_acc
                        saver.save(sess, 'model/{}'.format(cur_model_name))

    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))

    
####################################################################################################  
    
def kaggle_train_step(loss, learning_rate=1e-3):
    with tf.name_scope('kaggle_train_step'):
        step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return step

def predict(X_test,model_name):
    
    with tf.Session() as sess: 
        
        saver = tf.train.import_meta_graph('./model/%s'%model_name)
        saver.restore(sess, tf.train.latest_checkpoint('model/'))
        graph = tf.get_default_graph()
        
        tf_input = graph.get_operations()[0].name+':0'
        x = graph.get_tensor_by_name(tf_input) 
        print(x.shape)
        tf_input1 = graph.get_operations()[1].name+':0'
        y = graph.get_tensor_by_name(tf_input1) 
        print(y.shape)
        tf_input2 = graph.get_operations()[2].name+':0'
        proba = graph.get_tensor_by_name(tf_input2) 
        
        y_outputs = graph.get_operation_by_name('evaluate/ArgMax').outputs[0]
        print(y_outputs.shape)
        print(y)
        
        y_pred = np.zeros(3500)
        
        y_out = sess.run(y_outputs, feed_dict={x: X_test,y:y_pred,proba:0.95})
    return y_out
  