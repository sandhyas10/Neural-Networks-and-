#!/usr/bin/env/ python
# ECBM E4040 Fall 2017 Assignment 2
# TensorFlow CNN
#sources: cnn_sample.py
import tensorflow as tf
import numpy as np
import time
from ecbm4040.neuralnets.cnn_sample import *
from ecbm4040.image_generator import ImageGenerator
####################################
# TODO: Build your own LeNet model #
####################################
def my_LeNet(input_x, input_y,keep_prob,
          img_len=32, channel_num=3, output_size=10,
          conv_featmap=[6, 16], fc_units=[84],
          conv_kernel_size=[5, 5], pooling_size=[2, 2],
          l2_norm=0.01, seed=235):
    
    assert len(conv_featmap) == len(conv_kernel_size) and len(conv_featmap) == len(pooling_size)

   
    
   #norm layer
    norm_layer_0=norm_layer(input_x)
    
    
    #conv layer
    conv_layer_0 = conv_layer(input_x=norm_layer_0.output(),
                              in_channel=channel_num,
                              out_channel=conv_featmap[0],
                              kernel_shape=conv_kernel_size[0],
                              rand_seed=seed,index=0)
    
    #dropout layer
    dropout_layer_0=tf.nn.dropout(conv_layer_0.output(),keep_prob)
    #pooling layer
    pooling_layer_0 = max_pooling_layer(input_x=dropout_layer_0,
                                        k_size=pooling_size[0],
                                        padding="VALID")
    
    
    norm_layer_1=norm_layer(pooling_layer_0.output())
    conv_layer_1 = conv_layer(input_x=norm_layer_1.output(),
                              in_channel=conv_featmap[0],
                              out_channel=conv_featmap[1],
                              kernel_shape=conv_kernel_size[1],
                              rand_seed=seed,index=1)
    #dropout_layer_1=tf.nn.dropout(conv_layer_1.output(),proba)
    #print(dropout_layer_1.shape)
    pooling_layer_1 = max_pooling_layer(input_x=conv_layer_1.output(),
                                        k_size=pooling_size[1],
                                        padding="VALID")
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
    conv_w = [conv_layer_0.weight,conv_layer_1.weight]
    fc_w = [fc_layer_0.weight, fc_layer_1.weight,fc_layer_2.weight]

    # loss
    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
        l2_loss += tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in conv_w])

        label = tf.one_hot(input_y, 10)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=fc_layer_2.output()),
            name='cross_entropy')
        loss = tf.add(cross_entropy_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('LeNet_loss', loss)

    return fc_layer_2.output(), loss


####################################
#        End of your code          #
####################################

##########################################
# TODO: Build your own training function #
##########################################

def train_step_my(loss, learning_rate=1e-3):
    with tf.name_scope('train_step_my'):
        step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return step

def my_training(X_train, y_train, X_val, y_val, 
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
             pre_trained_model=None,prob=1.0):
    print("Building my LeNet. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))

    
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
        keep_prob=tf.placeholder(tf.float32)
    output, loss = my_LeNet(xs, ys,keep_prob,
                         img_len=32,
                         channel_num=3,
                         output_size=10,
                         conv_featmap=conv_featmap,
                         fc_units=fc_units,
                         conv_kernel_size=conv_kernel_size,
                         pooling_size=pooling_size,
                         l2_norm=l2_norm,
                         seed=seed)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    step = train_step_my(loss,learning_rate)
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
                
                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x, ys: training_batch_y,keep_prob:prob})

                if iter_total % 100 == 0:
                    # do validation
                    valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val, ys: y_val,keep_prob:1.0})
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


    pass
##########################################
#            End of your code            #
##########################################



def my_training_task4(X_train, y_train, X_val, y_val, 
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

    num_train=X_train.shape[0]
    
    X_train_reshaped=X_train.reshape([num_train,3,32,32]).transpose((0,2,3,1))
    X_val=X_val.reshape([X_val.shape[0],3,32,32]).transpose((0,2,3,1))
    img1=ImageGenerator(X_train_reshaped,y_train)
    img2=ImageGenerator(X_train_reshaped,y_train)
    img3=ImageGenerator(X_train_reshaped,y_train)
    img4=ImageGenerator(X_train_reshaped,y_train)
    
    img1.translate(16,16)
    img2.rotate(45.0)
    img3.flip("v")
    img4.add_noise(0.2,20)
              
    X_train=np.concatenate((img1.x,img2.x,img3.x,img4.x),axis=0)
    y_train=np.concatenate((img1.y,img2.y,img3.y,img4.y),axis=0)
    
   
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
        keep_prob=tf.placeholder(tf.float32)
        
        
    output, loss = my_LeNet(xs, ys,keep_prob,
                         img_len=32,
                         channel_num=3,
                         output_size=10,
                         conv_featmap=conv_featmap,
                         fc_units=fc_units,
                         conv_kernel_size=conv_kernel_size,
                         pooling_size=pooling_size,
                         l2_norm=l2_norm,
                         seed=seed)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    step = train_step_my(loss)
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
                if(training_batch_y.shape == 0):
                    print(itr)
                    print(training_batch_x.shape)
                    
    
                #print(y_val.shape)
                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x, ys: training_batch_y,keep_prob:pro})

                if iter_total % 100 == 0:
                    # do validation
                    valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val, ys: y_val,keep_prob:1.0})
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

    # TODO: Copy my_training function, make modifications so that it uses your data generator from task 4 to train.
