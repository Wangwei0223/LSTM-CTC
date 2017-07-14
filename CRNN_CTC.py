# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import matplotlib.pylab as plt
from tensorflow.contrib import legacy_seq2seq
import numpy as np
from tensorflow.contrib import rnn


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def con2d(x, W):  # 实现卷积和池化的两个函数 ， 因为是1步长，SAME补0,所以output和input尺寸一样
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def con2d_valid(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x): # [1, height, width, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def max_pool_1x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 2, 1, 1], padding='SAME')

'''
# sample 中是固定高度 31, 宽度不固定
# 灰度是一行矩阵，转成3行格式的才能输出
# image_raw_data = tf.gfile.FastGFile("/home/wangwei/sample/14/1/434_Filminess_28905.jpg", 'r').read()
image_raw_data = tf.gfile.FastGFile("/home/wangwei/3.jpg", 'r').read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    image = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    # print img_data.eval()
    print sess.run(image)
    print sess.run(tf.shape(img_data.eval()))

    #img = tf.image.resize_images(img_data, [10, 28])  # 二维10个，1维28个
    # print img.eval()

    #print sess.run(tf.shape(img.eval()))
    img = tf.image.rgb_to_grayscale(image)
    print sess.run(tf.shape(img.eval()))
    img = tf.image.grayscale_to_rgb(img)
    #print sess.run(tf.shape(img.eval()))
    plt.imshow(img.eval())
    plt.show()
'''

# x = tf.placeholder(tf.float32, [None, 32, None, 1])


# img = tf.Variable(tf.random_normal([32, 32, 100, 1]))


def CNN(x):

# conv1 = k:3*3, s:1, p:1 window 2*2
    W_conv1 = weight_variable([3, 3, 1, 64])
    b_conv1 = bias_variable([64])

    h_conv1 = tf.nn.relu(con2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

# conv2 = k:3*3, s:1, p:1 window 2*2
    W_conv2 = weight_variable([3, 3, 64, 128])
    b_conv2 = weight_variable([128])

    h_conv2 = tf.nn.relu(con2d(h_pool1, W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

# conv3 = k:3*3, s:1, p:1
    W_conv3 = weight_variable([3, 3, 128, 256])
    b_conv3 = weight_variable([256])

    h_conv3 = tf.nn.relu(con2d(h_pool2, W_conv3)+b_conv3)

# conv4 = k:3*3, s:1, p:1 window 1*2
    W_conv4 = weight_variable([3, 3, 256, 256])
    b_conv4 = weight_variable([256])

    h_conv4 = tf.nn.relu(con2d(h_conv3, W_conv4)+b_conv4)
    h_pool3 = max_pool_1x2(h_conv4)

# conv5 = k:3*3, s:1, p:1
    W_conv5 = weight_variable([3, 3, 256, 512])
    b_conv5 = weight_variable([512])

    h_conv5 = tf.nn.relu(con2d(h_pool3, W_conv5)+b_conv5)

# BatchNormalization
    # h_conv5 = tf.nn.batch_normalization(h_conv5)

# conv6 = k:3*3, s:1, p:1
    W_conv6 = weight_variable([3, 3, 512, 512])
    b_conv6 = weight_variable([512])

    h_conv6 = tf.nn.relu(con2d(h_conv5, W_conv6)+b_conv6)

# BatchNormalization
#    h_conv6 = tf.nn.batch_normalization(h_conv6)

    h_pool4 = max_pool_1x2(h_conv6)

# conv7 = k: 3*3, s:1, p:0
    W_conv7 = weight_variable([2, 2, 512, 512])
    b_conv7 = weight_variable([512])

    h_conv7 = tf.nn.relu(con2d_valid(h_pool4, W_conv7)+b_conv7)

# h_conv7 [128, 1, 6, 512]
# 降维为[batch_size, step*特征数]

    return tf.squeeze(h_conv7)


n_hidden = 256  # RNN隐藏层神经元
num_layers = 2
n_steps = 24
n_input = 512


def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), xrange(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def get_input_lens(sequences):
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
    return lengths


def BiRNN(x, weights, biases):

    # cell = rnn.BasicLSTMCell(num_units=n_hidden)
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

#   lstm_cell = rnn.MultiRNNCell([cell]*num_layers, state_is_tuple=True)

    # outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype=tf.float32)
    return tf.matmul(outputs[-1], weights) + biases


def RNN(input, seq_len):
    cell = tf.contrib.rnn.LSTMCell(n_hidden, state_is_tuple=True)

    # Stacking rnn cells
    stack = tf.contrib.rnn.MultiRNNCell([cell] * 2,
                                    state_is_tuple=True)

# The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, input, seq_len, dtype=tf.float32)

    return outputs

