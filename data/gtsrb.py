from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from . import data_info
import pickle                                                                    
import numpy as np                                                               
                                                                                 
                                                                                 
# Load pickled data                                                              
def load_data(data_dir=None, one_hot=False):
    """
    Load and preprocess GSTRB dataset
    :return: tuple of four arrays containing training data, training labels,
             testing data and testing labels.
    """
    if data_dir is not None:
        with open(data_dir + '/train.p', mode='rb') as f:                                    
            train = pickle.load(f)                                                   
        with open(data_dir + '/valid.p', mode='rb') as f:                                  
            valid = pickle.load(f)                                                   
        with open(data_dir + '/test.p', mode='rb') as f:                                     
            test = pickle.load(f)                                                    
                                                                                 
        X = {}
        Y = {}
        # split into train - test - validation
        X['train'] = train['features']
        Y['train'] = train['labels']
        X['test'] = test['features']
        Y['test'] = test['labels']
        X['val'] = valid['features']
        Y['val'] = valid['labels']

        # print shapes
        data_info.get_shapes(X, Y)

        return X, Y

    else:
        print('data_dir is None.. data not loaded')
        return None
