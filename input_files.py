import pandas as pd
import numpy as np
from tensorflow.contrib import learn


class InputHelper(object):

    field_test = []
    field_train = []
    path_test = ''
    path_train = ' '

    def __init__(self, path_test, path_train):

        # id - the id of a training set question pair
        # qid1, qid2 - unique ids of each question (only available in train.csv)
        # question1, question2 - the full text of each question
        # is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning

        self.field_train = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
        self.field_test = ['test_id', 'question1', 'question2']
        self.path_test = path_test
        self.path_train = path_train

    def get_size_q_train(self):

        df_train = pd.read_csv(self.path_train, skipinitialspace=True,usecols=self.field_train)

        size_max_q1 = 0
        size_max_q2 = 0

        for size_q1 in df_train['question1'].str.len():
            if size_q1 > size_max_q1:
                size_max_q1 = size_q1

        for size_q2 in df_train['question2'].str.len():
            if size_q2 > size_max_q2:
                size_max_q2 = size_q2

        return max(size_max_q1, size_max_q2)

    def get_size_q_test(self):

        df_test = pd.read_csv(self.path_test, skipinitialspace=True, usecols=self.field_test)

        size_max_q1 = 0
        size_max_q2 = 0

        for size_q1 in df_test['question1'].str.len():
            if size_q1 > size_max_q1:
                size_max_q1 = size_q1

        for size_q2 in df_test['question2'].str.len():
            if size_q2 > size_max_q2:
                size_max_q2 = size_q2

        return max(size_max_q1, size_max_q2)

    def get_voc_process(self):
        voc_process = learn.preprocessing.VocabularyProcessor(max(self.get_size_q_test(), self.get_size_q_train()),
                                                              min_frequency=0)
        return voc_process

    def get_train_data(self):

        df_train = pd.read_csv(self.path_train, skipinitialspace=True, usecols=self.field_train)
        voc_pr = self.get_voc_process()
        voc_pr.fit_transform(np.concatenate((np.array(df_train['question1'].tolist()),
                                             np.array(df_train['question2'].tolist())), axis=0))

        print(voc_pr.vocabulary_.__len__())

       # print(lista)

        return df_train

    def get_test_data(self):

        df_test = pd.read_csv(self.path_test, skipinitialspace=True, usecols=self.field_test)
        voc_pr = self.get_voc_process()
        voc_pr.transform(np.concatenate(df_test['question1'].values, df_test['question2'].values))
        return df_test

