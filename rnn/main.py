import input_files as input_file
import train_neural_network as train_nn
import numpy as np

MAX_QUESTIONS_SIZE = 128

data_questions = input_file.InputHelper('/home/rodoni/DataSets/QuoraQuestions/test.csv',
                              '/home/rodoni/DataSets/QuoraQuestions/train.csv', MAX_QUESTIONS_SIZE)

questions_1, questions_2, is_duplicated, vocab_size, max_document_size = data_questions.get_train_data()

rnn = train_nn.TrainNeuralNetworkSiamese(max_document_size, questions_1, questions_2, is_duplicated, vocab_size)
rnn.train_nn()
