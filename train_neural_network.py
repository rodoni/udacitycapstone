from neural_network import NeuralNetworkLSTM
import numpy as np
import tensorflow as tf


class TrainNeuralNetworkSiamese(object):

    def __init__(self, max_doc_size, questions_1, questions_2, is_duplicate, vocab_size):
        self.max_doc_size = max_doc_size
        self.questions_1 = questions_1
        self.questions_2 = questions_2
        self.is_duplicated = is_duplicate
        self.vocab_size = vocab_size

    def train_nn(self):

        r_nn = NeuralNetworkLSTM(self.max_doc_size, self.vocab_size, (self.max_doc_size*10))
        i_questions_1 = iter(self.questions_1)
        i_questions_2 = iter(self.questions_2)
        i_is_duplicate = iter(self.is_duplicated)
        sess = tf.Session()

        feed_dict = {
            r_nn.question_1: i_questions_1.next(),
            r_nn.question_2: i_questions_2.next(),
            r_nn.is_duplicate: i_is_duplicate.next(),
            r_nn.dropout_keep_prob: 0.5
        }

        sess.run([r_nn.run_siamese_network()], feed_dict)












