import input_files as input_file

data_questions = input_file.InputHelper('/home/rodoni/DataSets/QuoraQuestions/test.csv',
                              '/home/rodoni/DataSets/QuoraQuestions/train.csv')

data_questions.get_train_data()



