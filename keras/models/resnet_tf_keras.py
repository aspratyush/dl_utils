'''
Building keras based model for mnist for testing
KerasModelWrapper in cleverhans
'''

from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.applications import resnet50

def model(nb_classes=10, logits=False, input_ph=None, nb_rows=28, nb_cols=28, nb_channels=1):

    # load the resnet50 model
    model = resnet50.ResNet50(include_top=False, weights='imagenet')
    print('---------> Shape : ', model.output_shape[1:])

    # define the top-model
    top_model = Sequential()

    layers = [Flatten(input_shape=model.output_shape[1:]),
            Dense(120, activation='relu'),
            Dense(84, activation='relu'),
            Dense(nb_classes)]

    # add layers to the model
    for layer in layers:
        top_model.add(layer)

    # add top_model to model
    model.add(top_model)

    # check if logits need to be returned
    if logits == True and input_ph is not None:
        logits_tensor = model(input_ph)

    # add softmax
    model.add(Activation('softmax'))

    if logits == True and input_ph is not None:
        return model, logits_tensor
    else:
        return model
