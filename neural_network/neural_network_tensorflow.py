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
        assert len(input_data)==len(output_data) and batch_size!=0
        self._examples_num=len(input_data)
        self._batch_size=batch_size
        self._batch_idx=0
        self._input_data=input_data
        self._output_data=output_data
        self._data_set_end=False

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def data_set_end(self):
        return self._data_set_end

    def all_samples(self):
        return self._input_data, self._output_data

    def next_batch(self):
        if self._batch_idx==0:
            self._data_set_end=False
        start=self._batch_idx*self._batch_size
        end=min((self._batch_idx+1)*self._batch_size,self._examples_num)
        self._batch_idx=(self._batch_idx+1)%(math.ceil(self._examples_num/self._batch_size))
        if end==self._examples_num:
            self._data_set_end=True
        return self._input_data[start:end], self._output_data[start:end]

class NeuralNetworkTensorflow(object):
    def __init__(self, sizes,act_func="sigmoid",output_func=None):
        #sizes表示神经网络各层的神经元个数，第一层为输入层，最后一层为输出层
        assert len(sizes)>1
        self.layer_sizes=sizes
        self._sess=None
        self._input_placeholder=None
        self._logits=None

        if act_func=="leaky_relu":
            self._act_func=tf.nn.leaky_relu
        elif act_func=="relu":
            self._act_func=tf.nn.relu
        elif act_func=="tanh":
            self._act_func=tf.nn.tanh
        else:
            self._act_func=tf.nn.sigmoid

        if output_func is None:
            self._output_func=None
        elif output_func=="relu":
            self._output_func=tf.nn.relu
        elif output_func=="leaky_relu":
            self._output_func=tf.nn.leaky_relu
        elif output_func=="tanh":
            self._output_func=tf.nn.tanh
        else:
            self._output_func=tf.nn.sigmoid

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
                    active_values = self._act_func(tf.matmul(active_values, weights) + biases)
            else:
                # Linear
                with tf.name_scope('softmax_linear'):
                    weights = tf.Variable(
                        tf.truncated_normal([last_layer_size, current_layer_size],
                                            stddev=1.0 / math.sqrt(float(last_layer_size))),
                        name='weights')
                    biases = tf.Variable(tf.zeros([current_layer_size]),
                                         name='biases')
                    if self._output_func is None:
                        logits = tf.matmul(active_values, weights) + biases
                    else:
                        logits = self._output_func(tf.matmul(active_values, weights) + biases)
        return logits

    def loss(self,logits,output_placeholder):
        if self.layer_sizes[-1]>1:
            #labels_sum=tf.expand_dims(tf.reduce_sum(output_placeholder),1)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(\
                logits=logits, labels=output_placeholder, name='xentropy')
                #logits=logits, labels=output_placeholder/labels_sum, name='xentropy')
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

    #评估模型
    def evaluate(self,test_data_set,evaluate_func):
        if evaluate_func is None:
            return
        output_result_batch=[]
        while True:
            input_data,output_data=test_data_set.next_batch()
            feed_dict={self._input_placeholder:input_data}
            result=self._sess.run(self._logits,feed_dict=feed_dict)
            output_result_batch.append((result,output_data))
            if test_data_set.data_set_end:
                break
        evaluate_func(output_result_batch)

    #训练模型
    def train(self, training_data_sets, epochs, learning_rate,evaluate_func=None):
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
                    print('Epoch %d: loss=%.3f (%.3f sec)'%(epoch+1,loss_value,duration))
                    self.evaluate(training_data_sets.test,evaluate_func)

def get_data_sets(x,y,batch_size):
    class DataSets(object):
        def __init__(self):
            pass
    data_sets=DataSets()
    sample_num=len(x)
    assert sample_num==len(y)
    shuffle_batch_idxs=np.arange(sample_num)
    np.random.shuffle(shuffle_batch_idxs)
    input_data=x[shuffle_batch_idxs]
    output_data=y[shuffle_batch_idxs]
    train_data_set=DataSet(input_data[:int(sample_num*0.9)],output_data[:int(sample_num*0.9)],batch_size)
    data_sets.train=train_data_set
    test_data_set=DataSet(input_data[int(sample_num*0.9):],output_data[int(sample_num*0.9):],batch_size)
    data_sets.test=test_data_set
    return data_sets

if __name__ == "__main__":
    #其中第一层为输入层，最后一层为输出层
    network=NeuralNetworkTensorflow([1,512,256,128,64,32,16,8,4,1],act_func="tanh")

    #训练集样本
    x = np.array([np.linspace(-7, 7, 500)]).T
    #训练集结果
    y = np.cos(x)*2

    training_data_sets=get_data_sets(x,y,100)

    #迭代5000次；学习率设为0.02
    network.train(training_data_sets,5000,0.02)

    #测试集样本
    x_test = np.array([np.linspace(-12, 12, 150)]).T
    #测试集结果
    y_predict = network.predict(x_test)

    #图示对比训练集和测试集数据
    plt.plot(x,y,'r',x_test,y_predict,'*')
    plt.show()