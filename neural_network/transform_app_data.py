#coding=utf-8
import argparse
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import re
import jieba
import os
import csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default='mlg')
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

def main(args):
    # x_original=title_raw.select("title").rdd.flatMap(lambda x: x).collect()
    x_raw = ["国务院任免工作人员：盛来运任国家统计局副局长","我国成功发射沙特-5A/5B卫星 搭载发射10颗小卫星","昔日斗鱼一姐沦落陪玩，看到接单数那一刻，众人纷纷感叹！"]
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

    args.checkpoint_dir=os.path.join(os.getcwd(),"model")

    # Map data into vocabulary
    vocab_path = os.path.join(args.checkpoint_dir, "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_splited)))
    #print(x_test)

    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(args.checkpoint_dir)
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

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                print("batch_predictions:",batch_predictions)
                all_predictions = np.concatenate([all_predictions, batch_predictions])
                print("all_predictions:",all_predictions)

    # Save the evaluation
    predictions = np.column_stack((np.array(x_raw), all_predictions))
    print("predictions:",predictions)

if __name__ == '__main__':
    args = parse_args()
    main(args)
