from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob

def load_data(data_dir, one_hot=False):
    """
    Helper function to load dogscats dataset
    :param data_dir : top-level directory containing the data
    :param one_hot : whether to one-hot encode the data
    """

    X = {}
    Y = {}
    if data_dir is not None:

        modes = ['train', 'valid']
        classes = ['cats', 'dogs']

    
        for mode in modes:
            X[mode] = []
            Y[mode] = []
    
            for tr_class in classes:
                print('-----------', mode, tr_class, '------------')
                file_path = data_dir + '/' + mode + '/' + tr_class + '/*.jpg'
                print(file_path)
                files = glob.glob(file_path)
                X[mode] += (files)
                Y[mode] += ([classes.index(tr_class)]*len(files))
                print(len(X[mode]), len(Y[mode]))

    else:
        print('data_dir is None...')
        
    return X, Y
