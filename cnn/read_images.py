import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math


class InputImages(object):

    def __init__(self, path, max_image, format_img,n_batch):
        self.path = path
        self.max_image = max_image
        self.format_img = format_img
        self.num_batches = n_batch

    def get_data(self, num_image):

        #print('Loading image', num_image)
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


        return image

    def get_batch(self, batch):

        size_batch = self.max_image / self.num_batches
        size_batch = math.ceil(size_batch)
        start_image = batch*size_batch
        last_image = start_image + size_batch
        image = start_image
        images = []

        while image < last_image:

            if image < self.max_image:
                image_matrix = self.get_data(image)
                images.append(image_matrix)
                image = image + 1
            else:
                break

        return images, start_image, image - 1



