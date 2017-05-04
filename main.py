import input_files as input_file
import numpy as np


data_questions = input_file.InputHelper('/home/rodoni/DataSet/Quora/test.csv',
                              '/home/rodoni/DataSet/Quora/train.csv')

data_questions.get_train_data()
