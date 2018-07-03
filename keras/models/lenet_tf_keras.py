'''
Building keras based model for mnist for testing
KerasModelWrapper in cleverhans
'''

from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense


def model(nb_classes=10, logits=False, input_ph=None, nb_rows=28, nb_cols=28, nb_channels=1):

    model = Sequential()

    layers = [Conv2D(6, (5,5), activation='relu', input_shape=(nb_rows,nb_cols,nb_channels), padding='valid'),
                MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'),
                Conv2D(16, (5,5), activation='relu', padding='valid'),
                MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'),
                Flatten(),
                Dense(120, activation='relu'),
                Dense(84, activation='relu'),
                Dense(nb_classes)]

    # add layers to the model
    for layer in layers:
        model.add(layer)

    # check if logits need to be returned
    if logits == True and input_ph is not None:
        logits_tensor = model(input_ph)

    # add softmax
    model.add(Activation('softmax'))

    if logits == True and input_ph is not None:
        return model, logits_tensor
    else:
        return model
