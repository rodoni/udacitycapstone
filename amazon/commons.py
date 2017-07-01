
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

    thres = [0.07, 0.17, 0.2, 0.04, 0.23, 0.33, 0.24, 0.22, 0.1, 0.19, 0.23, 0.24, 0.12, 0.14, 0.25, 0.26, 0.16]
    #thres = [0.2, 0.15, 0.2, 0.45, 0.23, 0.33, 0.24, 0.22, 0.1, 0.19, 0.8, 0.24, 0.12, 0.14, 0.25, 0.4, 0.16]






