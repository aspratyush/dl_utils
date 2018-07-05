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

            print('filenames, labels = ', len(filenames), ' ', len(labels))

    else:
        print('data_dir/modes/classes is None...')

    return filenames, labels


def get_img_generator(input_fn_dict, fn_get_data=get_filenames_and_labels):
    """
    returns the data_generator for this dataset.
    need to get filenames to pass to td.data.Dataset
    :param fn_get_data : functor that returns filenames and labels
    :param input_fn_dict : dictionary containing the following:
        :param data_dir : path to the data, used by fn_get_data
        :param input_name : model input layer name
        :param modes : list of modes train/valid/test
        :param classes : list of classes
        :param nb_rows : height of the image
        :param nb_cols : width of the image
        :param nb_channels : channels in the image
        :param nb_buffers : buffer size for shuffling
        :param nb_repeats : number of times to repeat the dataset
        :param nb_batches : batches of images to return per run
        :param one_hot : whether to one-hot encode the data
    """
    # get constants from the dictionary
    print('inside data_generator, input_fn_dict : ', input_fn_dict)
    data_dir = input_fn_dict['data_dir']
    input_name = input_fn_dict['input_name']
    modes = input_fn_dict['modes']
    classes = input_fn_dict['classes']
    nb_batches = input_fn_dict['nb_batches']
    nb_buffers = input_fn_dict['nb_buffers']
    nb_repeats = input_fn_dict['nb_repeats']
    nb_channels = input_fn_dict['nb_channels']
    nb_rows = input_fn_dict['nb_rows']
    nb_cols = input_fn_dict['nb_cols']
    one_hot = input_fn_dict['one_hot']

    # get the filenames and labels
    filenames, labels = get_filenames_and_labels(data_dir, modes, classes)
    # get tensors from inputs
    filenames = tf.constant(filenames)
    labels = tf.constant(labels)
    labels = tf.cast(labels, tf.int32)

    # inline function to parse all filenames and labels
    def parse_fn(filename, label):
        # read
        img_str = tf.read_file(filename)
        # decode
        img = tf.image.decode_image(img_str, channels=nb_channels)
        img.set_shape([None, None, None])
        # convert to float
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize
        img = tf.image.resize_images(img, [nb_rows, nb_cols])
        img.set_shape([nb_rows, nb_cols, nb_channels])

        # one-hot label
        if (one_hot is True):
            label = tf.one_hot(label, len(classes))

        # dictionary
        d = dict(zip([input_name], [img])), label

        return d

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
