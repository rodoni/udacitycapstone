import pandas as pd
import numpy as np
from tensorflow.contrib import learn
import re

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",
                          re.UNICODE)


def tokenizer(iterator):
    for value in iterator:
        yield list(value)


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

        df_train = pd.read_csv(self.path_train, skipinitialspace=True, usecols=self.field_train)

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

    def get_train_data(self):

        """Read the train.csv file and convert it in a vocabulary and numbers that represents
           each word and returns all questions in arrays with numbers inside, together with questions
           we are returning the if the question has or not a duplicate meaning """
        max_document_size = int(max(self.get_size_q_train(), self.get_size_q_test()))

        print("Reading CSV files")

        df_train = pd.read_csv(self.path_train, skipinitialspace=True, usecols=self.field_train)

        print("Creating Vocabulary Pre Processor")
        voc_pr = learn.preprocessing.VocabularyProcessor(max_document_size)

        questions_1 = df_train['question1'].tolist()
        questions_2 = df_train['question2'].tolist()
        questions = pd.concat([df_train['question1'].dropna(), df_train['question2'].dropna()])

        voc_pr.fit(questions)

        print("Size of vocabulary loaded = {}".format(len(voc_pr.vocabulary_)))

        questions1 = voc_pr.transform(questions_1)
        questions2 = voc_pr.transform(questions_2)

        is_duplicate = np.array(df_train['is_duplicate'].tolist())

        print("Vocabulary parsed")

        return questions1, questions2, is_duplicate, len(voc_pr.vocabulary_), max_document_size

    def get_test_data(self):

        """Read the test.csv file and convert it in a vocabulary and numbers that represents
           each word and returns all questions in arrays with numbers inside """

        print("Reading CSV files")

        df_test = pd.read_csv(self.path_test, skipinitialspace=True, usecols=self.field_test)

        print("Creating Vocabulary Pre Processor")
        voc_pr = learn.preprocessing.VocabularyProcessor(int(max(self.get_size_q_train(), self.get_size_q_test())))

        questions_1 = df_test['question1'].tolist()
        questions_2 = df_test['question2'].tolist()
        questions = pd.concat([df_test['question1'].dropna(), df_test['question2'].dropna()])

        voc_pr.fit(questions)

        print("Size of vocabulary loaded = {}".format(len(voc_pr.vocabulary_)))

        questions1 = voc_pr.transform(questions_1)
        questions2 = voc_pr.transform(questions_2)

        print("Vocabulary parsed")

        return questions1, questions2

