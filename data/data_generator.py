from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import tensorflow as tf


def get_filenames_and_labels(data_dir, modes, classes):
    """
    :param data_dir : dir containing the images
    :param modes : list of modes of the training
    :param classes : list of classes in the dataset
    """
    if data_dir is not None and modes is not None and classes is not None:

        for mode in modes:
            filenames = []
            labels = []

            for tr_class in classes:
                print('-----------', mode, tr_class, '------------')
                file_path = data_dir + '/' + mode + '/' + tr_class + '/*.jpg'
                print(file_path)
                files = glob.glob(file_path)
                filenames += (files)
                labels += ([classes.index(tr_class)]*len(files))

    else:
        print('data_dir/modes/classes is None...')

    return filenames, labels


def get_img_generator(
                        fn_get_data=get_filenames_and_labels, data_dir=None,
                        input_name='input_1', modes=None, classes=None,
                        nb_rows=224, nb_cols=224,
                        nb_channels=3, nb_classes=10,
                        nb_buffers=256, nb_repeats=1,
                        nb_batches=128, one_hot=False):
    """
    returns the data_generator for this dataset.
    need to get filenames to pass to td.data.Dataset
    :param fn_get_data : functor that returns filenames and labels
    :param data_dir : path to the data
    :param input_name : model input layer name
    :param modes : list of modes train/valid/test
    :param classes : list of classes
    :param nb_rows : height of the image
    :param nb_cols : width of the image
    :param nb_channels : channels in the image
    :param nb_classes : number of classes in the dataset
    :param nb_buffers : buffer size for shuffling
    :param nb_repeats : number of times to repeat the dataset
    :param nb_batches : batches of images to return per run
    :param one_hot : whether to one-hot encode the data
    """
    # inline function to parse all filenames and labels
    def parse_fn(filename, label):
        print(filename, label)
        # read
        img_str = tf.read_file(filename)
        # decode
        img = tf.image.decode_image(img_str, channels=3)
        # convert to float
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize
        img.set_shape([None, None, None])
        img = tf.image.resize_images(img, [nb_rows, nb_cols])

        # one-hot label
        if (one_hot is True):
            label = tf.one_hot(label, len(classes))

        # dictionary
        d = dict(zip(['input_1'], [img])), label

        return d

    filenames, labels = get_filenames_and_labels(data_dir, modes, classes)
    # get tensors from inputs
    filenames = tf.constant(filenames)
    labels = tf.constant(labels)
    labels = tf.cast(labels, tf.int32)

    # tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # perform map fn on the dataset to read and resize the images
    dataset = dataset.map(parse_fn)

    # add batch and shuffle information
    dataset = dataset.shuffle(nb_buffers)
    dataset = dataset.repeat(nb_repeats)
    dataset = dataset.batch(nb_batches)

    # make iterator
    it = dataset.make_one_shot_iterator()
    batch_features, batch_labels = it.get_next()

    return batch_features, batch_labels
