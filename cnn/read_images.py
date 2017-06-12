import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


class InputImages(object):

    def __init__(self, path, max_image, format_img):
        self.path = path
        self.max_image = max_image
        self.format_img = format_img

    def get_data(self):

        num_image = 0
        list_images = []
        while num_image <= self.max_image:
            name_file = "train_"+str(num_image)+"."+str(self.format_img)
            num_image = num_image + 1
            img = mpimg.imread(self.path+name_file)

            """
            Normalization for jpg image with 4 channels 
            it will be different when use other format 
            """
            image = img.astype('float32')
            image = np.array(image/255.0)
            image = np.delete(image, 3, axis=2)
            list_images.append(image)

        return list_images
