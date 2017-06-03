import read_files as rf
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


path_to_files = sys.argv[1]
train_file = path_to_files+"train.csv"
train_image_dir = path_to_files+"/train/"
test_image_dir = path_to_files+"/test/"

if len(sys.argv) < 1:
    print('Please put the arguments')
    exit
else:

    if sys.argv[1] == "--help":
        print('Please use the the follow sintaxe: script <path_to_files>')
        exit


read_file = rf.InputFileReader(train_file)
activity_data, weather_data = read_file.get_data()

# load image
print(activity_data)
img = mpimg.imread(train_image_dir+"train_1000.jpg")
print(img.shape)

#plt.imshow(img, cmap='gray', interpolation='bicubic')
#plt.show()
# normalization

image = img.astype('float32')
image = np.array(image/255.0)
# reshape image
image = np.delete(image, 3, axis=2)
print(image.shape)
print(image)
