import read_files as rf
import read_images as ri
import sys
import numpy as np
import pandas as pd

# import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


# global variables
path_to_files = sys.argv[1]
train_file = path_to_files+"train.csv"
train_image_dir = path_to_files+"train/"
test_image_dir = path_to_files+"test/"
N_OF_BATCHES = 1000
N_OF_CLASS = 12

# check input variable
if len(sys.argv) < 1:
    print('Please put the arguments')
    exit
else:

    if sys.argv[1] == "--help":
        print('Please use the the follow sintaxe: script <path_to_files>')
        exit

#read label
read_file = rf.InputFileReader(train_file)
activity_data, weather_data, index = read_file.get_data()

print("Max Index : "+str(index))
num_row = 0

#while num_row < index:
#    row = np.array(activity_data.iloc[num_row:num_row+1].values)
#    row = np.delete(row, 0)
#    num_row = num_row + 1

#read features
read_img = ri.InputImages(train_image_dir, index, "jpg", N_OF_BATCHES)

# Create the model
model = Sequential()
model.add(Conv2D(32, (4, 4), input_shape=(256, 256, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (4, 4), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(16, 16)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(N_OF_CLASS, activation='softmax'))

# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


# features and labels train
images_array, start_image, end_image = read_img.get_batch(0)
rows = np.array(activity_data.iloc[start_image:end_image+1].values)
x_features = np.array(images_array)
y_labels = np.delete(rows, 0, axis=1)

print("Amount of Images : ", x_features.shape[0])
print("Start Image", start_image)
print("End Image", end_image)

i_train = 0

model.train_on_batch(x_features, y_labels)

# features and labels test
images_array, start_image, end_image = read_img.get_batch(250)
rows = np.array(activity_data.iloc[start_image:end_image+1].values)
x_features = np.array(images_array)
y_labels = np.delete(rows, 0, axis=1)

# Final evaluation of the model
scores = model.evaluate(x_features, y_labels, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#while num_image < index:
#    image = read_img.get_data(num_image)
#    print("Load Image : "+str(num_image))
#    num_image = num_image + 1

