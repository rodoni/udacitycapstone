import tensorflow as tf
from tensorflow.contrib import rnn


class NeuralNetworkLSTM(object):

    """
    A LSTM based in a deep siamese neural network for text similarity
    """

    def __init__(self, sequence_length, vocab_size, embedding_size):
        self.question_1 = tf.placeholder(tf.int32, [None, sequence_length], name="question_1")
        self.question_2 = tf.placeholder(tf.int32, [None, sequence_length], name="question_2")
        self.is_duplicate = tf.placeholder(tf.int32, [None], name="is_duplicate")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.n_steps = sequence_length
        self.vocab_size = vocab_size
        self.n_input = embedding_size
        self.distance = 0
        self.accuracy = 0

    def run_siamese_network(self):

        weight = tf.Variable(tf.random_uniform([self.vocab_size, self.n_input], 0, 1), trainable=True, name="weight")

        embedded_chars_1 = tf.nn.embedding_lookup(weight, self.question_1)
        embedded_chars_2 = tf.nn.embedding_lookup(weight, self.question_2)

        out_1 = self.neural_net_rnn(embedded_chars_1)
        out_2 = self.neural_net_rnn(embedded_chars_2)

        self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(out_1, out_2)), 1, keep_dims=True))
        self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keep_dims=True)),
                                                     tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keep_dims=True))))

        self.distance = tf.reshape(self.distance, [-1], name="distance")

        correct_prediction = tf.equal(self.distance, self.is_duplicate)

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")

    def neural_net_rnn(self, data):

        n_layers = 3

        #data = tf.transpose(data, [1, 0, 2])
        #data = tf.reshape(data, [-1, self.n_input])
        #data = tf.split(data, self.n_steps, 0)
        print(data)

        """ Define lstm cells with tensorflow """

        """Forward direction """

        fw_cell = rnn.core_rnn_cell.BasicLSTMCell(self.n_steps, forget_bias=1.0, state_is_tuple=True, reuse=True)
        #lstm_fw_cell = rnn.core_rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
        #lstm_fw_cell_m = rnn.core_rnn_cell.MultiRNNCell([lstm_fw_cell]*n_layers, state_is_tuple=True)

        """Backward direction """

        bw_cell = rnn.core_rnn_cell.BasicLSTMCell(self.n_steps, forget_bias=1.0, state_is_tuple=True, reuse=True)
        #lstm_bw_cell = rnn.core_rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
        #lstm_bw_cell_m = rnn.core_rnn_cell.MultiRNNCell([lstm_bw_cell]*n_layers, state_is_tuple=True)

        out_puts, state, _ = rnn.static_bidirectional_rnn(fw_cell,
                                                          bw_cell,
                                                          tf.unstack(tf.transpose(data, perm=[1, 0, 2])),
                                                          dtype=tf.float32)

        return out_puts[-1]












