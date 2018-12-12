#coding=utf-8
import os
import pickle
import numpy as np
from neural_network_tensorflow import DataSet,NeuralNetworkTensorflow,get_data_sets

def train_model(train_data_x,train_data_y):
    #其中第一层为输入层，最后一层为输出层
    network=NeuralNetworkTensorflow([28,1024,512,1024,256,5],\
        act_func="leaky_relu",output_func="relu")

    training_data_sets=get_data_sets(train_data_x,train_data_y,4096)

    #迭代2000次；学习率设为0.01
    network.train(training_data_sets,2000,0.01,evaluate_func=eva_func)

def eva_func(result_batch):
    positive_false=0
    positive_num=0
    negetive_false=0
    negetive_num=0
    for result,output_data in result_batch:
        for sample_idx,result_sample in enumerate(result):
            result_max_idx=np.argmax(result_sample)
            output_sample=output_data[sample_idx]
            output_max_idx=np.argmax(output_sample)
            if output_max_idx==4:
                negetive_num+=1
                if result_max_idx!=output_max_idx:
                    negetive_false+=1
            else:
                if output_sample[output_max_idx]==1:
                    positive_num+=1
                    if result_max_idx!=output_max_idx and output_sample[result_max_idx]!=1:
                        positive_false+=1
    print("positive_false=%d/%d=%.5f negetive_false=%d/%d=%.5f"\
        %(positive_false,positive_num,positive_false/positive_num,\
        negetive_false,negetive_num,negetive_false/negetive_num))

def main():
    train_data_x_pkl_file=os.path.join(os.getcwd(),"x.pkl")
    train_data_y_pkl_file=os.path.join(os.getcwd(),"y.pkl")

    if not os.path.exists(train_data_x_pkl_file) or not os.path.exists(train_data_y_pkl_file):
        print("train data not found!")
    else:
        with open(train_data_x_pkl_file, 'rb') as f:
            train_data_x = pickle.load(f)
            print("read pkl:",train_data_x_pkl_file)
        with open(train_data_y_pkl_file, 'rb') as f:
            train_data_y = pickle.load(f)
            print("read pkl:",train_data_y_pkl_file)

    train_model(train_data_x,train_data_y)

if __name__ == '__main__':
    main()
