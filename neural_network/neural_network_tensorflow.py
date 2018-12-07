#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pylab as plt
import random
import math
import tensorflow as tf
import time

class DataSet(object):
    def __init__(self,input_data,output_data,batch_size):
        assert len(input_data)==len(output_data)
        assert len(input_data)%batch_size==0 and batch_size!=0
        self._examples_num=len(input_data)
        self._batch_size=batch_size
        self._batch_idx=0
        shuffle_batch_idxs=np.arange(self._examples_num)
        np.random.shuffle(shuffle_batch_idxs)
        self._input_data=input_data[shuffle_batch_idxs]
        self._output_data=output_data[shuffle_batch_idxs]

    @property
    def batch_size(self):
        return self._batch_size

    def next_batch(self):
        start=self._batch_idx*self._batch_size
        end=(self._batch_idx+1)*self._batch_size
        self._batch_idx=(self._batch_idx+1)%(self._examples_num//self._batch_size)
        return  self._input_data[start:end], self._output_data[start:end]

class NeuralNetworkTensorflow(object):
    def __init__(self, sizes):
        #sizes表示神经网络各层的神经元个数，第一层为输入层，最后一层为输出层
        assert len(sizes)>1
        self.layer_sizes=sizes
        self._sess=None
        self._input_placeholder=None
        self._logits=None
        #self.act_func=tf.nn.leaky_relu
        self.act_func=tf.nn.tanh

    def inference(self,input_placeholder):
        active_values=input_placeholder
        for size_idx in range(1,len(self.layer_sizes)):
            last_layer_size=self.layer_sizes[size_idx-1]
            current_layer_size=self.layer_sizes[size_idx]
            if size_idx<len(self.layer_sizes)-1:
                # Hidden
                with tf.name_scope("hidden"+str(size_idx)):
                    weights = tf.Variable(
                        tf.truncated_normal([last_layer_size, current_layer_size],
                                            stddev=1.0 / math.sqrt(float(last_layer_size))),
                        name='weights')
                    biases = tf.Variable(tf.zeros([current_layer_size]),
                                         name='biases')
                    active_values = self.act_func(tf.matmul(active_values, weights) + biases)
            else:
                # Linear
                with tf.name_scope('softmax_linear'):
                    weights = tf.Variable(
                        tf.truncated_normal([last_layer_size, current_layer_size],
                                            stddev=1.0 / math.sqrt(float(last_layer_size))),
                        name='weights')
                    biases = tf.Variable(tf.zeros([current_layer_size]),
                                         name='biases')
                    logits = tf.matmul(active_values, weights) + biases
        return logits

    def loss(self,logits,output_placeholder):
        if self.layer_sizes[-1]>1:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(\
                logits=logits, labels=output_placeholder, name='xentropy')
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        else:
            distance=tf.square(tf.subtract(logits,output_placeholder))
            loss=tf.reduce_mean(distance)
        return loss

    #预测结果
    def predict(self, input_data):
        feed_dict={self._input_placeholder:input_data}
        result=self._sess.run(self._logits,feed_dict=feed_dict)
        return result

    #训练模型
    def train(self, training_data_sets, epochs, learning_rate):
        with tf.Graph().as_default():
            input_placeholder = tf.placeholder(tf.float32,
                shape=(None, self.layer_sizes[0]))
                #shape=(training_data_sets.train.batch_size, self.layer_sizes[0]))
            self._input_placeholder=input_placeholder

            output_placeholder = tf.placeholder(tf.float32,
                shape=(None, self.layer_sizes[-1]))
                #shape=(training_data_sets.train.batch_size, self.layer_sizes[-1]))

            logits=self.inference(input_placeholder)
            self._logits=logits

            loss=self.loss(logits,output_placeholder)

            # Add a scalar summary for the snapshot loss.
            #tf.summary.scalar(loss.op.name, loss)
            # Create the gradient descent optimizer with the given learning rate.
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            # Create a variable to track the global step.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            # Use the optimizer to apply the gradients that minimize the loss
            # (and also increment the global step counter) as a single training step.
            train_op = optimizer.minimize(loss, global_step=global_step)

            #sess=tf.Session()
            init=tf.initialize_all_variables()
            sess=tf.Session()
            self._sess=sess
            sess.run(init)
            for epoch in range(epochs):
                start_time=time.time()

                input_data,output_data=training_data_sets.train.next_batch()
                feed_dict={
                    input_placeholder:input_data,
                    output_placeholder:output_data
                }
                _,loss_value=sess.run([train_op,loss],feed_dict=feed_dict)

                duration=time.time()-start_time

                if epoch%100 == 99:
                    print('Epoch %d: loss=%.2f (%.3f sec)'%(epoch+1,loss_value,duration))

def get_data_sets(x,y,batch_size):
    class DataSets(object):
        def __init__(self):
            pass
    data_sets=DataSets()
    train_data_set=DataSet(x,y,batch_size)
    data_sets.train=train_data_set
    return data_sets

if __name__ == "__main__":
    #其中第一层为输入层，最后一层为输出层
    network=NeuralNetworkTensorflow([1,512,256,128,64,32,16,8,4,1])

    #训练集样本
    x = np.array([np.linspace(-7, 7, 400)]).T
    #训练集结果
    y = np.cos(x)

    training_data_sets=get_data_sets(x,y,100)

    #迭代5000次；学习率设为0.02
    network.train(training_data_sets,5000,0.02)

    #测试集样本
    x_test = np.array([np.linspace(-9, 9, 80)]).T
    #测试集结果
    y_predict = network.predict(x_test)

    #图示对比训练集和测试集数据
    plt.plot(x,y,'r',x_test,y_predict,'*')
    plt.show()