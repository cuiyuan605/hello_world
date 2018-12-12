#coding=utf-8
import argparse
import tensorflow as tf
from tensorflow.contrib import learn
import pandas as pd
import numpy as np
import re
import jieba
import os
import pickle



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='D:\\github\\hello_world\\neural_network\\model\\')
    parser.add_argument("--data_path", type=str, default='D:\\github\\hello_world\\neural_network\\20181210_2\\')
    args = parser.parse_args()
    return args


def sp_title(title):
    rule = '[^/.0-9a-zA-Z\u4e00-\u9fa5]+'
    c = re.sub(rule, '', title)
    sp = jieba.lcut(c)
    if len(sp) < 3:
        return ''
    res = ''
    for s in sp:
        res += str(s) + ' '
    return res

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def title2class(news_title,checkpoint_dir):
    x_raw=news_title
    x_splited=[]
    for title in x_raw:
        x_splited.append(sp_title(title))
    #print(x_splited)

    # Eval Parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    #
    FLAGS = tf.flags.FLAGS

    # Map data into vocabulary
    vocab_path = os.path.join(checkpoint_dir, "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_splited)))
    #print(x_test)

    print("\nEvaluating...\n")

    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for idx,x_test_batch in enumerate(batches):
                print("%d/%d..."%(idx,len(news_title)//64))
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    return all_predictions

def convert_ad_list(ad_list_str):
    ad_list=[]
    for ad_id in ad_list_str.split('|'):
        if ad_id=="2":
            ad_list.append(0)
        elif ad_id=="3":
            ad_list.append(1)
        elif ad_id=="4":
            ad_list.append(2)
        elif ad_id=="100010":
            ad_list.append(3)
        elif ad_id=="100016":
            ad_list.append(1)
        elif ad_id=="100017":
            ad_list.append(0)
    return ad_list

def get_user_click_data(data_path):
    click_data=[]
    news_title=[]
    title_count=0

    if not os.path.isdir(data_path):
        print("wrong dir:",data_path)
        exit()

    for file_name in os.listdir(data_path):
        file_path=os.path.join(data_path,file_name)
        with open(file_path,'r',encoding='UTF-8') as f:
            file_lines=f.readlines()
            print(file_path,"lines:",len(file_lines))
            for idx,line in enumerate(file_lines):
                if idx<1:
                    continue
                line=line.strip('\r\n ')
                values=line.split('\t',6)
                if len(values)<7:
                    print("error line:",values)
                    break
                data_row=[]
                #devicetype
                data_row.append(int(values[1]))
                #brand
                data_row.append(int(values[2]))
                #screen width
                data_row.append(int(values[3]))
                #ad_click_list
                data_row.append(convert_ad_list(values[4]))
                #ad_not_click_list
                data_row.append(convert_ad_list(values[5]))
                #title list
                title_list=[t for t in values[6].split('|') if t!='']
                news_title+=title_list
                data_row.append(len(title_list))
                title_count+=len(title_list)
                click_data.append(data_row)

    assert title_count==len(news_title),"title_count error %d"%title_count
    return click_data,news_title

def add_input_data(input_data,sample_value,num):
    assert sample_value<num
    for idx in range(num):
        if idx==sample_value:
            input_data.append(1)
        else:
            input_data.append(0)

def add_title_data(input_data,title_class_list):
    tilte_data_list=[0 for idx in range(10)]
    for title_class in title_class_list:
        tilte_data_list[int(title_class)]+=1
    input_data+=tilte_data_list

def add_output_data(output_data,ad_click_list,not_click_list):
    ctrs=[0.0160,0.0383,0.0156,0.0352]
    ad_num=len(ctrs)
    for idx in range(ad_num):
        if idx in ad_click_list:
            output_data.append(1)
        elif idx in not_click_list:
            output_data.append(0)
        else:
            output_data.append(ctrs[idx])
    if output_data.count(0)==ad_num:
        output_data.append(1)
    else:
        output_data.append(0)

def convert_to_train_data(click_data,news_title_class):
    train_input_data=[]
    train_output_data=[]
    title_idx=0
    for sample in click_data:
        input_data=[]
        add_input_data(input_data,sample[0],3)
        add_input_data(input_data,sample[1],11)
        add_input_data(input_data,sample[2],4)
        add_title_data(input_data,news_title_class[title_idx:title_idx+sample[5]])
        title_idx+=sample[5]
        train_input_data.append(input_data)

        output_data=[]
        add_output_data(output_data,sample[3],sample[4])
        train_output_data.append(output_data)

    return np.array(train_input_data),np.array(train_output_data)

def main(args):
    train_data_x_pkl_file=os.path.join(os.getcwd(),"x.pkl")
    train_data_y_pkl_file=os.path.join(os.getcwd(),"y.pkl")

    if os.path.exists(train_data_x_pkl_file):
        os.remove(train_data_x_pkl_file)
    if os.path.exists(train_data_y_pkl_file):
        os.remove(train_data_y_pkl_file)

    #获取用户点击新闻的title数据
    click_data,news_title=get_user_click_data(args.data_path)

    #根据title获取新闻的分类
    all_classes=title2class(news_title,args.model_dir)

    assert len(news_title)==len(all_classes)

    train_data_x,train_data_y=convert_to_train_data(click_data,all_classes)

    with open(train_data_x_pkl_file, 'wb') as f:
        pickle.dump(train_data_x, f)
        print("write pkl:",train_data_x_pkl_file)
    with open(train_data_y_pkl_file, 'wb') as f:
        pickle.dump(train_data_y, f)
        print("write pkl:",train_data_y_pkl_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
