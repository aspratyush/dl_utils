'''
Building keras based model for mnist for testing
KerasModelWrapper in cleverhans
'''

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense


def model(nb_classes, logits=False):

    model = Sequential()

    '''
    layers = [Conv2D(6, (5,5), activation='relu', input_shape=(28,28,1)),
                MaxPooling2D(pool_size=(2,2), strides=(2,2),
                Conv2D(16, (5,5), activation='relu'),
                MaxPooling2D(pool_size=(2,2), strides=(2,2)),
                Flatten,
                Dense(120, activation='relu'),
                Dense(84, activation='relu')]
    '''

    model.add(Conv2D(6,(5,5),activation='relu', input_shape=(32,32,3), padding='VALID'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='VALID'))
    model.add(Conv2D(16, (5,5), activation='relu', padding='VALID'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='VALID'))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(nb_classes))
    #model.add(Activation('softmax'))

    if logits == True:
        return logits, model
    else:
        return model
