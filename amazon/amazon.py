import numpy as np
import pandas as pd
import read_files as rf
import cnn_arquiteture as cnn
import commons as cm
import sys

# global variables
N_MAX_BATCH = 1
MAX_IMAGES_TO_TRAIN = 40478
MAX_IMAGES_TO_TEST = 61191
IMAGE_RES = [64, 64]


data_set_path = ""
train_and_predict = True


# Check input variable
if len(sys.argv) > 1:
    data_set_path = sys.argv[1]
    if len(sys.argv) == 3:
        if sys.argv[2] == '--only-predict':
            print('If will search by weights of one pre-trained network')
            train_and_predict = False


else:
    print('Please use the the follow syntax: script <path_to_data_set>')
    exit(1)

train_file = data_set_path+"train.csv"
test_file = data_set_path+"sample_submission.csv"

train_path = data_set_path+"train/"
test_path = data_set_path+"test/"

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

com_variables = cm.CommonsVariables()
data_image = rf.InputImagesTrain(train_path, MAX_IMAGES_TO_TRAIN, IMAGE_RES[0], IMAGE_RES[1], df_train, N_MAX_BATCH)
data_test = rf.InputImagesTest(test_path, MAX_IMAGES_TO_TEST, IMAGE_RES[0], IMAGE_RES[1], df_test, N_MAX_BATCH)

# Train
if train_and_predict:
    batch = 0
    x_train, y_train = data_image.get_data_image(batch)
    print(x_train.shape)
    c_net = cnn.ConvNet()
    threshold = c_net.neural_net_train(x_train, y_train, 5)

# Prediction
batch_test = 0
output = []
name_images = []

x_test, y_label = data_test.get_data_image(batch_test)
name_images.extend(y_label)
c_net = cnn.ConvNet()
output.extend(c_net.neural_net_predict_fold(x_test, 5))
batch_test = batch_test + 1

output = np.array(output)
data_result = pd.DataFrame(output, columns=com_variables.labels)
file_result = pd.DataFrame(index=range(MAX_IMAGES_TO_TEST), columns=['image_name', 'tags'])

# Creating submission file to kaggle
print('Creating file to submission ...')
for i in range(data_result.shape[0]):
    data = data_result.ix[[i]]
    data = data.apply(lambda x: x > 0.23, axis=1)
    data = data.transpose()
    data = data.loc[data[i] == True]
    ' '.join(list(data.index))
    file_result.set_value(i, 'tags', ' '.join(list(data.index)))
    file_result.set_value(i, 'image_name', name_images[i])

# Save results in CSV format
file_result.to_csv('submission_keras.csv', index=False)
data_result.to_csv('data_result.csv', index=False)


exit(0)

