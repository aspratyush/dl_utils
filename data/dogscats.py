from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import glob
import matplotlib.pyplot as plt
from . import data_info
from .preprocessing import transforms

def load_data(data_dir, one_hot=False):
    """
    Helper function to load dogscats dataset
    :param data_dir : top-level directory containing the data
    :param one_hot : whether to one-hot encode the data
    """

    if data_dir is not None:
        # Training
        classes = ['dogs', 'cats']

        X = {}
        Y = {}
        for item in classes:
            # get the path
            path = data_dir + "/train/" + item + "/*.jpg"
            
            # get all the files
            train_files = glob.glob(path)
            print(path, train_files[0])
            
            # test with 1st image
            img = plt.imread(train_files[0])
            img_new = transforms.scale_min(img, 224)

    return img, img_new

