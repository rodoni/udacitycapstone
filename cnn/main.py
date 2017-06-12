import read_files as rf
import read_images as ri
import sys
import numpy as np
from keras.models import Sequential


path_to_files = sys.argv[1]
train_file = path_to_files+"train.csv"
train_image_dir = path_to_files+"train/"
test_image_dir = path_to_files+"test/"

if len(sys.argv) < 1:
    print('Please put the arguments')
    exit
else:

    if sys.argv[1] == "--help":
        print('Please use the the follow sintaxe: script <path_to_files>')
        exit

#read image information
read_file = rf.InputFileReader(train_file)
activity_data, weather_data, index = read_file.get_data()
print ("Max Index : "+str(index))

read_img = ri.InputImages(train_image_dir, index, "jpg")
num_image = 2

while num_image <= 10000:
    image = read_img.get_data(num_image)
    print("Load Image : "+str(num_image))
    num_image = num_image + 1

