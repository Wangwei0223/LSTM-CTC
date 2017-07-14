# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import os
import preprocessimage as pr
import CRNN_CTC as model
from tensorflow.contrib import rnn

# parameter
sample_file = os.path.abspath('..') + "/sample/sample.txt"

LETTERS = "abcdefghijklmnopqrstuvwxyz"
num_classes = len(LETTERS) + 1  # 26 digits + ctc blank
batch_size = 32
learning_rate = 0.001
MOMENTUM = 0.9
n_hidden = 256  # RNN隐藏层神经元 与模型中的对应

# parameter

train_path, label_list = pr.read_data_set(sample_file)

num_examples = 6400

test_path = train_path[0:16]
test_path_label = label_list[0:16]

num_batches_per_epoch = int(num_examples/batch_size)

img = tf.Variable(tf.random_normal([batch_size, 32, 100, 1]))  # 32*100通道1

x = tf.placeholder(tf.float32, [None, 32, 100, 1])  # 到时的输入input

y = model.CNN(x)

W = tf.Variable(tf.truncated_normal(shape=[n_hidden, num_classes], stddev=0.1))

b = tf.Variable(tf.constant(0.1, shape=[num_classes]))

# Here we use sparse_placeholder that will generate a
# SparseTensor required by ctc_loss op.
targets = tf.sparse_placeholder(tf.int32)  # 到时的输入label，需要用稀疏矩阵

seq_len = tf.placeholder(tf.int32, [None])  # 1d array of size [batch_size] 说白了32个512[512, 512,..., 512, 512]

inputs = tf.placeholder(tf.float32, [None, None, 512])

cell = tf.contrib.rnn.LSTMCell(n_hidden, state_is_tuple=True)


# Stacking rnn cells
stack = tf.contrib.rnn.MultiRNNCell([cell] * 2,
                                    state_is_tuple=True)

# The second output is the last state and we will no use that
outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

shape = tf.shape(inputs)
batch_s, max_timesteps = shape[0], shape[1]

# Reshaping to apply the same weights over the timesteps
outputs = tf.reshape(outputs, [-1, n_hidden])

logits = tf.matmul(outputs, W) + b

logits = tf.reshape(logits, [batch_s, -1, num_classes])

logits = tf.transpose(logits, (1, 0, 2))  # [24, 32, 27]

label = pr.preprocess_label(label_list)
test_label = pr.preprocess_label(test_path_label)
train_targets = label
test_targets = test_label

seq = np.ones([batch_size], dtype=int) * 24

loss = tf.nn.ctc_loss(targets, logits, seq_len)

cost = tf.reduce_mean(loss)

# optimizer = tf.train.MomentumOptimizer(learning_rate=1e-2, momentum=0.9).minimize(cost)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss)

# Option 2: tf.contrib.ctc.ctc_beam_search_decoder
# (it's slower but you'll get better results)
# decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
# Inaccuracy: label error rate
ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                      targets))

init = tf.global_variables_initializer()


def train_network():
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver(tf.all_variables())
        path = os.path.abspath('..') + "/Model_CRNN 2/CRNN_MNIST_Image.ckpt"
        for curr_epoch in range(250):
            train_cost = train_ler = 0
            for batch in range(num_batches_per_epoch):
                indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
                sparse_targets = model.sparse_tuple_from_label(train_targets[indexes])   # 处理完的标签数据 稀疏矩阵
                process_list = np.asarray(train_path)[indexes]
                resized = pr.preprocess_img(process_list, 32, 100)

                cnn_output = sess.run(y, feed_dict={x: sess.run(resized)})

                # print "cnn output:", sess.run(tf.shape(cnn_output))

                feed = {inputs: cnn_output,
                        targets: sparse_targets,
                        seq_len: seq}

                # print "rnn output:", sess.run(tf.shape(logits), feed_dict=feed)  # rnn output: [ 24 32 27]

                batch_cost, _ = sess.run([cost, optimizer], feed)
                train_cost += batch_cost * batch_size
                train_ler += sess.run(ler, feed_dict=feed)*batch_size

                print ("Iter:", batch, "batch_cost: {:.6f}".format(batch_cost))

            save_path = saver.save(sess, path, global_step=curr_epoch)
            print("Model saved in file: ", save_path)
            # Metrics mean
            train_cost /= num_examples
            train_ler /= num_examples
            log = "cur_epoch = {:.1f}, train_cost = {:.3f}, train_ler = {:.3f}"
            print(log.format(curr_epoch, train_cost, train_ler))


def test_network():
    with tf.Session() as sess:
        saver = tf.train.Saver()
        path = os.path.abspath('..') + "/Model_CRNN 2/CRNN_MNIST_Image.ckpt-4"
        saver.restore(sess, path)
        batch_test_target = model.sparse_tuple_from_label(test_targets)
        process_test_list = np.asarray(test_path)
        resized2 = pr.preprocess_img(process_test_list, 32, 100)

        cnn_output2 = sess.run(y, feed_dict={x: sess.run(resized2)})

        print ("CNN output", sess.run(tf.shape(cnn_output2)))

        batch_test_seq_len = np.ones([16], dtype=int) * 24
        feed = {inputs: cnn_output2,
                targets: batch_test_target,
                seq_len: batch_test_seq_len
                }
        # Decoding
        d = sess.run(decoded[0], feed_dict=feed)
        dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=sess)

        for i, seq in enumerate(dense_decoded):

            seq = [s for s in seq if s != -1]

            # print('Sequence %d' % i)
            print('\nOriginal:\t%s' % test_targets[i])

            for z in range(len(test_targets[i])):
                print(chr(test_targets[i][z]+97), end='')

            print('\nDecoded:\t%s' % seq)
            for z in range(len(seq)):
                print(chr(seq[z]+97), end='')

            print('\n')

train_network()