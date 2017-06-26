import numpy as np
import math
import pandas as pd
from tqdm import tqdm
import cv2
import commons as cm


class InputImagesTrain(object):

    def __init__(self, path, max_images, resize_w, resize_h, train_file, num_batches):

        '''
        :param path: path for the image files
        :param max_images: max number of images inside of directory 
        :param resize_w: resize of images 
        :param resize_h: resize of images 
        :param train_file: csv file with information about classification 
        :param num_batches: number of batches desired 
        '''

        self.path = path
        self.max_images = max_images
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.train_file = train_file
        self.num_batches = num_batches

    def get_data_image(self, batch):

        size_batch = self.max_images / self.num_batches
        size_batch = math.ceil(size_batch)
        start_image = int(batch * size_batch)
        last_image = int(start_image + size_batch)
        if last_image > self.max_images:
            last_image = self.max_images

        x_images = []
        y_labels = []

        train_file = pd.DataFrame(self.train_file)
        com_variables = cm.CommonsVariables()

        for f, tags in tqdm(train_file.values[start_image:last_image], miniters=1000):

            img = cv2.imread('{}{}.jpg'.format(self.path, f))
            targets = np.zeros(17)
            for t in tags.split(' '):
                targets[com_variables.label_map[t]] = 1
            x_images.append(cv2.resize(img, (self.resize_w, self.resize_h)))
            y_labels.append(targets)

        y_labels = np.array(y_labels, np.uint8)
        x_images = np.array(x_images, np.float32) / 255.

        return x_images, y_labels


class InputImagesTest(object):

    def __init__(self, path, max_images, resize_w, resize_h, test_file, num_batches):

        '''
        :param path: path for the image files
        :param max_images: max number of images inside of directory
        :param resize_w: resize of images
        :param resize_h: resize of images
        :param test_file: csv file with information about classification
        :param num_batches: number of batches desired
        '''

        self.path = path
        self.max_images = max_images
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.test_file = test_file
        self.num_batches = num_batches

    def get_data_image(self, batch):

        size_batch = self.max_images / self.num_batches
        size_batch = math.ceil(size_batch)
        start_image = int(batch * size_batch)
        last_image = int(start_image + size_batch)
        if last_image > self.max_images:
            last_image = self.max_images

        x_images = []
        y_labels = []

        test_file = pd.DataFrame(self.test_file)

        for f, tags in tqdm(test_file.values[start_image:last_image], miniters=1000):

            img = cv2.imread('{}{}.jpg'.format(self.path, f))
            x_images.append(cv2.resize(img, (self.resize_w, self.resize_h)))
            y_labels.append(f)

        x_images = np.array(x_images, np.float32) / 255.

        return x_images, y_labels
