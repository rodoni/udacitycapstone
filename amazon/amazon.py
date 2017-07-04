import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential

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


com_variables = cm.CommonsVariables()
data_image = rf.InputImagesTrain('C:/DataSet/Amazon/train/', MAX_IMAGES_TO_TRAIN, 64, 64, df_train, N_MAX_BATCH)
data_test = rf.InputImagesTest('C:/DataSet/Amazon/test/', MAX_IMAGES_TO_TEST, 64, 64, df_test, N_MAX_BATCH)



batch = 0
while batch < N_MAX_BATCH:

    x_train, y_train = data_image.get_data_image(batch)
    print(x_train.shape)
    c_net = cnn.ConvNet()
    model = Sequential()
    model = c_net.neural_net_train(x_train, y_train, 5)
    batch = batch + 1

batch_test = 0

output = []
name_images = []
while batch_test < N_MAX_BATCH:

    x_test, y_label = data_test.get_data_image(batch_test)
    name_images.extend(y_label)
    c_net = cnn.ConvNet()
    output.extend(c_net.neural_net_predict_fold(x_test, 5))
    batch_test = batch_test + 1

output = np.array(output)
data_result = pd.DataFrame(output, columns=com_variables.labels)

file_result = pd.DataFrame(index=range(MAX_IMAGES_TO_TEST), columns=['image_name', 'tags'])


for i in range(data_result.shape[0]):
    data = data_result.ix[[i]]
    data = data.apply(lambda x: x > 0.2, axis=1)
    data = data.transpose()
    data = data.loc[data[i] == True]
    ' '.join(list(data.index))
    file_result.set_value(i, 'tags', ' '.join(list(data.index)))
    file_result.set_value(i, 'image_name', name_images[i])


file_result.to_csv('submission_keras.csv', index=False)
data_result.to_csv('submission_values.csv', index=False)


exit(0)

