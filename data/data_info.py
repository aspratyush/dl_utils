from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def get_shapes(X, Y):
    """
    get_shapes : get the shapes of the dictionaries passed
    :param X : dictionary of train / test / validation features
    :param Y : dictionary of train / test / validation labels
    """

    template = '{} shape : {}'
    
    # check keys are same in X and Y
    keys = [keys for keys in X.keys() if keys in Y.keys()]

    for key in keys:
        print(template.format('X_' + key, X[key].shape))
        print(template.format('Y_' + key, Y[key].shape))

        
