import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc


import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

import cv2
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score
import time

import read_files as rf
import ccn_arquiteture as cnn
import commons as cm

x_train = []
x_test = []
y_train = []
N_MAX_BATCH = 1
MAX_IMAGES_TO_TRAIN = 40478
MAX_IMAGES_TO_TEST = 61191

df_train = pd.read_csv('D:/DataSet/Amazon/train.csv')
df_test = pd.read_csv('D:/DataSet/Amazon/sample_submission.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

labels = ['blow_down',
          'bare_ground',
          'conventional_mine',
          'blooming',
          'cultivation',
          'artisinal_mine',
          'haze',
          'primary',
          'slash_burn',
          'habitation',
          'clear',
          'road',
          'selective_logging',
          'partly_cloudy',
          'agriculture',
          'water',
          'cloudy']

label_map = {'agriculture': 14,
             'artisinal_mine': 5,
             'bare_ground': 1,
             'blooming': 3,
             'blow_down': 0,
             'clear': 10,
             'cloudy': 16,
             'conventional_mine': 2,
             'cultivation': 4,
             'habitation': 9,
             'haze': 6,
             'partly_cloudy': 13,
             'primary': 7,
             'road': 11,
             'selective_logging': 12,
             'slash_burn': 8,
             'water': 15}

com_variables = cm.CommonsVariables()
data_image = rf.InputImagesTrain('D:/DataSet/Amazon/train/', MAX_IMAGES_TO_TRAIN, 32, 32, df_train, N_MAX_BATCH)
data_test = rf.InputImagesTest('D:/DataSet/Amazon/test/', MAX_IMAGES_TO_TEST, 32, 32, df_test, N_MAX_BATCH)



batch = 0
while batch < N_MAX_BATCH:

    x_train, y_train = data_image.get_data_image(batch)
    c_net = cnn.ConvNet()
    model = Sequential()
    model = c_net.neural_net_train(x_train, y_train, batch)
    batch = batch + 1

batch_test = 0

output = []
name_images = []
while batch_test < N_MAX_BATCH:

    x_test, y_label = data_test.get_data_image(batch_test)
    name_images.extend(y_label)
    c_net = cnn.ConvNet()
    output.extend(c_net.neural_net_predict(x_test, batch - 1))
    batch_test = batch_test + 1

output = np.array(output)
data_result = pd.DataFrame(output, columns=com_variables.labels)

file_result = pd.DataFrame(index=range(MAX_IMAGES_TO_TEST), columns=['image_name', 'tags'])


for i in range(data_result.shape[0]):
    data = data_result.ix[[i]]
    data = data.apply(lambda x: x > com_variables.thres, axis=1)
    data = data.transpose()
    data = data.loc[data[i] == True]
    ' '.join(list(data.index))
    file_result.set_value(i, 'tags', ' '.join(list(data.index)))
    file_result.set_value(i, 'image_name', name_images[i])


file_result.to_csv('submission_keras.csv', index=False)
data_result.to_csv('submission_values.csv', index=False)


exit(0)

