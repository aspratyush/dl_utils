'''
Inspired from:
    1) Keras blog : simple Conv-net : https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
    2) Keras blog : fine-tuning VGG : https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = '/home/ctg_pratyush/workspace/data/dogscats/sample/train'
validation_data_dir = '/home/ctg_pratyush/workspace/data/dogscats/sample/valid'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 20
batch_size = 16

def main():

    finetune = True

    if (finetune == True):

        print('Downloading Resnet...')
        # model
        #prev_model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))
        prev_model = keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))

        # top model
        top_model = Sequential()
        top_model.add(Flatten(input_shape=prev_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(84, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(1, activation='sigmoid'))

        # model summary
        top_model.summary()

        # set prev_model to be non-trainable
        for layer in prev_model.layers:
            layer.trainable = False

        # append the models
        model = Model(inputs=prev_model.input, outputs=top_model(prev_model.output))



    else:
        # Model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

    # model summary
    model.summary()

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    for layer in model.layers:
        print(layer, layer.trainable)


    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)


if __name__ == '__main__':
    main()
