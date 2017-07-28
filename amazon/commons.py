import cv2
import numpy as np


class CommonsVariables(object):

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

    def satured_image_color(self, img, s_factor):

        """
        :param img: img in numpy matrix
        :param s_factor: multiplication factor for saturation
        :return: 
        """
        # load file in hsv format
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv = img_hsv.astype("float32")
        # modify saturation
        (h, s, v) = cv2.split(img_hsv)
        s = s*s_factor
        s = np.clip(s, 0, 255)
        img_hsv_s = cv2.merge([h, s, v])
        img_rgb_s = cv2.cvtColor(img_hsv_s.astype("uint8"), cv2.COLOR_HSV2BGR)

        return img_rgb_s






