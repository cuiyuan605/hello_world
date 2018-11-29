from neural_network_2 import NeuralNetwork2
import numpy as np
import matplotlib.pylab as plt
import random

#距离函数的偏导数
def distance_derivative(output_activations, y):
    #损失函数的偏导数
    return 2*(output_activations-y)

# sigmoid函数
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# sigmoid函数的导数
def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

def ouput_layer_func(z):
    return z

def ouput_layer_func_derivative(z):
    return z-z+1

if __name__ == "__main__":
    #其中第一层为输入层，最后一层为输出层
    network=NeuralNetwork2([1,256,256,128,128,64,64,32,32,16,16,8,8,4,4,1],sigmoid,sigmoid_derivative,
        distance_derivative,ouput_layer_func,ouput_layer_func_derivative)

    #训练集样本
    x = np.array([np.linspace(-7, 7, 200)]).T
    #训练集结果，由于使用了sigmoid作为激活函数，需保证其结果落在(0,1)区间内
    y = np.cos(x)

    #使用随机梯度下降算法（SGD）对模型进行训练
    #迭代5000次；每次随机抽取40个样本作为一个batch；学习率设为0.1
    training_data=[(np.array([x_value]),np.array([y_value])) for x_value,y_value in zip(x,y)]
    network.SGD(training_data,5000,40,0.1)

    #测试集样本
    x_test = np.array([np.linspace(-9, 9, 120)])
    #测试集结果
    y_predict = network.feedforward(x_test)

    #图示对比训练集和测试集数据
    plt.plot(x,y,'r',x_test.T,y_predict.T,'*')
    plt.show()