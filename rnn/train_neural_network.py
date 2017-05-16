
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


class TrainNeuralNetworkSiamese(object):

    def __init__(self, max_doc_size, questions_1, questions_2, is_duplicate, vocab_size):
        self.max_doc_size = max_doc_size
        self.questions_1 = questions_1
        self.questions_2 = questions_2
        self.is_duplicated = is_duplicate
        self.vocab_size = vocab_size
        self.display_step = 10

    def train_nn(self):

        question_1_tf = tf.placeholder("float", [None, 1, self.max_doc_size])
        question_2_tf = tf.placeholder("float", [None, 1, self.max_doc_size])
        is_duplica_tf = tf.placeholder("float", [None, 1])

        # Define weights
        weights = {
            # Hidden layer weights => 2*n_hidden because of forward + backward cells
            'out': tf.Variable(tf.random_normal([2 * self.max_doc_size, self.vocab_size]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.vocab_size]))
        }

        output_1 = self.bi_rnn(question_1_tf, weights, biases)

        # Define loss and optimizer

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_1, labels=is_duplica_tf))
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(cost)

        # Evaluate Model
        correct_pred = tf.equal(tf.argmax(output_1, 1), tf.argmax(is_duplica_tf, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()

        question_1_iter = iter(self.questions_1)
        is_duplica_iter   = iter(self.is_duplicated)

        with tf.Session() as sess:
            sess.run(init)
            step = 1

            while step < 11:
                question_1 = question_1_iter.next()
                is_duplica = is_duplica_iter.next()

                #Run optimizer backprop
                sess.run(optimizer, feed_dict={question_1_tf: question_1, is_duplica_tf: is_duplica})
                if step % self.display_step == 0:
                    acc = sess.run(accuracy,feed_dict={question_1_tf: question_1, is_duplica_tf: is_duplica})

                    loss = sess.run(cost, feed_dict={question_1_tf: question_1, is_duplica_tf: is_duplica})

                    print("Loss " + "{:.6f}".format(loss) + " Accuracy" + "{:.5f}".format(acc))

                step += 1

    def bi_rnn(self, question_1_tf, weights, biases):

        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(question_1_tf, self.max_doc_size, 0)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(self.max_doc_size, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(self.max_doc_size, forget_bias=1.0)

        # Get lstm cell output

        try:
                outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                             dtype=tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
                outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                       dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']












