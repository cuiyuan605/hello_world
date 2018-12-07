#coding=utf-8
import tensorflow as tf
import numpy as np

sess=tf.Session()

#input_placeholder=tf.placeholder(tf.int32, shape=(2, 3))
#input_data=np.array([[4,5,6],[7,8,9]])
#feed_dict={input_placeholder:input_data}

#batch_size = tf.size(input_placeholder)
#result=sess.run(batch_size,feed_dict=feed_dict)

#input_data_expand_dims = tf.expand_dims(input_placeholder, 1)
#result=sess.run(input_data_expand_dims,feed_dict=feed_dict)

#indices = tf.expand_dims(tf.range(0, batch_size), 1)
#concated = tf.concat(1, [indices, input_data_expand_dims])
#result=sess.run(concated,feed_dict=feed_dict)

batch_size=5
class_num=3

logits=tf.placeholder(tf.float32, shape=(batch_size, class_num))
input_data=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
output=tf.placeholder(tf.int32, shape=(batch_size))
output_data=np.array([0,2,1,0,2])
feed_dict={logits:input_data,output:output_data}

result=sess.run(logits,feed_dict=feed_dict)
print("logits:",result)
result=sess.run(output,feed_dict=feed_dict)
print("output:",result)

batch_size = tf.size(output)
result=sess.run(batch_size,feed_dict=feed_dict)
print("batch_size:",result)

labels = tf.expand_dims(output, 1)
result=sess.run(labels,feed_dict=feed_dict)
print("labels:",result)

indices = tf.expand_dims(tf.range(0, batch_size), 1)
result=sess.run(indices,feed_dict=feed_dict)
print("indices:",result)

concated = tf.concat([indices, labels],1)
result=sess.run(concated,feed_dict=feed_dict)
print("concated:",result)

shape=tf.stack([batch_size,class_num])
result=sess.run(shape,feed_dict=feed_dict)
print("shape:",result)

onehot_labels = tf.sparse_to_dense(
    concated, shape, 1.0, 0.0)
result=sess.run(onehot_labels,feed_dict=feed_dict)
print("onehot_labels:",result)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                           labels=onehot_labels,
                                                           name='xentropy')
result=sess.run(cross_entropy,feed_dict=feed_dict)
print("cross_entropy:",result)

loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
result=sess.run(loss,feed_dict=feed_dict)
print("loss:",result)

sess.close()