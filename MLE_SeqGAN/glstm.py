# '''
# A Dynamic Recurrent Neural Network (LSTM) implementation example using
# TensorFlow library. This example is using a toy dataset to classify linear
# sequences. The generated sequences have variable length.
#
# Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
#
# Author: Aymeric Damien
# Project: https://github.com/aymericdamien/TensorFlow-Examples/
# '''
#
# from __future__ import print_function
#
import tensorflow as tf

# import random

#
# # ====================
# #  TOY DATA GENERATOR
# # ====================
# class ToySequenceData(object):
#     """ Generate sequence of data with dynamic length.
#     This class generate samples for training:
#     - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
#     - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
#
#     NOTICE:
#     We have to pad each sequence to reach 'max_seq_len' for TensorFlow
#     consistency (we cannot feed a numpy array with inconsistent
#     dimensions). The dynamic calculation will then be perform thanks to
#     'seqlen' attribute that records every actual sequence length.
#     """
#
#     def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
#                  max_value=1000):
#         self.data = []
#         self.labels = []
#         self.seqlen = []
#         for i in range(n_samples):
#             # Random sequence length
#             len = random.randint(min_seq_len, max_seq_len)
#             # Monitor sequence length for TensorFlow dynamic calculation
#             self.seqlen.append(len)
#             # Add a random or linear int sequence (50% prob)
#             if random.random() < .5:
#                 # Generate a linear sequence
#                 rand_start = random.randint(0, max_value - len)
#                 s = [[float(i) / max_value] for i in
#                      range(rand_start, rand_start + len)]
#                 # Pad sequence for dimension consistency
#                 s += [[0.] for i in range(max_seq_len - len)]
#                 self.data.append(s)
#                 self.labels.append([1., 0.])
#             else:
#                 # Generate a random sequence
#                 s = [[float(random.randint(0, max_value)) / max_value]
#                      for i in range(len)]
#                 # Pad sequence for dimension consistency
#                 s += [[0.] for i in range(max_seq_len - len)]
#                 self.data.append(s)
#                 self.labels.append([0., 1.])
#         self.batch_id = 0
#
#     def next(self, batch_size):
#         """ Return a batch of data. When dataset end is reached, start over.
#         """
#         if self.batch_id == len(self.data):
#             self.batch_id = 0
#         batch_data = (self.data[self.batch_id:min(self.batch_id +
#                                                   batch_size, len(self.data))])
#         batch_labels = (self.labels[self.batch_id:min(self.batch_id +
#                                                       batch_size, len(self.data))])
#         batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
#                                                       batch_size, len(self.data))])
#         self.batch_id = min(self.batch_id + batch_size, len(self.data))
#         return batch_data, batch_labels, batch_seqlen


# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_iters = 1000000
batch_size = 128
display_step = 2

# Network Parameters
seq_max_len = 2  # Sequence max length
n_ele_dim = 300
n_hidden = 64  # hidden layer num of features
n_classes = 2  # linear sequence or not

# trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
# testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)

import numpy as np

homicide_data = np.load('homicide_in_num.npy')
protest_data = np.load('protest_in_num.npy')
data = np.concatenate((homicide_data, protest_data)).tolist()
label = np.zeros([homicide_data.shape[0] + protest_data.shape[0], n_classes])
label[:homicide_data.shape[0], 0] = 1
label[homicide_data.shape[0]:, 1] = 1
label = label.tolist()

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, n_ele_dim])
y = tf.placeholder("float", [None, n_classes])

# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


class glstm(object):
    def __init__(self, x, seqlen, weights, biases):
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_ele_dim])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, seq_max_len, x)

        self.x = x

        # Define a lstm cell with tensorflow
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

        # Get lstm cell output, providing 'sequence_length' will perform dynamic
        # calculation.
        outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32,
                                    sequence_length=seqlen)

        # When performing dynamic calculation, we must retrieve the last
        # dynamically computed output, i.e., if a sequence length is 10, we need
        # to retrieve the 10th output.
        # However TensorFlow doesn't support advanced indexing yet, so we build
        # a custom op that for each sample in batch size, get its length and
        # get the corresponding relevant output.

        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, n_step, n_input]
        outputs = tf.pack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])

        # Hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

        # Linear activation, using outputs computed above
        self.g_predictions = tf.matmul(outputs, weights['out']) + biases['out']

        # Define loss and optimizer
        self.pretrain_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.g_predictions, y))

        # Evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.g_predictions, 1), tf.argmax(y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.pretrain_loss)


tlstm = glstm(x, seqlen, weights, biases)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations


    while step * batch_size < training_iters:
        # batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        batch_x = data[(step-1)*batch_size:step*batch_size]
        batch_y = label[(step-1)*batch_size:step*batch_size]
        batch_seqlen = np.full((batch_size), 2, dtype=int).tolist()

        if len(batch_x) != batch_size:
            print 'help me'
            break

        sess.run(tlstm.optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(tlstm.accuracy, feed_dict={x: batch_x, y: batch_y,
                                                      seqlen: batch_seqlen})
            # Calculate batch loss
            loss = sess.run(tlstm.pretrain_loss, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # # Calculate accuracy
    # test_data = testset.data
    # test_label = testset.labels
    # test_seqlen = testset.seqlen
    # print("Testing Accuracy:", \
    #       sess.run(tlstm.accuracy, feed_dict={x: test_data, y: test_label,
    #                                           seqlen: test_seqlen}))
