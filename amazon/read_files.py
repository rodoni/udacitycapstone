import Image
import numpy as np
import math
import pandas as pd
from tqdm import tqdm
import cv2


class InputImages(object):
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

    def get_images(self, batch):

        size_batch = self.max_images / self.num_batches
        size_batch = math.ceil(size_batch)
        start_image = int(batch * size_batch)
        last_image = int(start_image + size_batch)
        if last_image > self.max_images:
            last_image = self.max_images

        x_images = []
        y_labels = []

        train_file = pd.DataFrame(self.train_file)

        for f, tags in tqdm(train_file.values[start_image:last_image], miniters=1000):

            img = cv2.imread('{}{}.jpg'.format(self.path, f))
            targets = np.zeros(17)
            for t in tags.split(' '):
                targets[self.label_map[t]] = 1
            x_images.append(cv2.resize(img, (self.resize_w, self.resize_h)))
            y_labels.append(targets)

        y_labels = np.array(y_labels, np.uint8)
        x_images = np.array(x_images, np.float32) / 255.

        return x_images, y_labels
