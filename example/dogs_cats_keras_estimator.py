from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
from dl_utils.keras.models import lenet_tf_keras
import dl_utils.data.data_generator as dg
import dl_utils.tf.estimators.keras_estimator as est_keras

# constants
nb_rows, nb_cols, nb_channels = 224, 224, 3

# input_fn_dict
input_fn_dict = {}
eval_fn_dict = {}
input_fn_dict['data_dir'] = '/mnt/data/Personal/coursera/DL/data/dogscats/sample'
input_fn_dict['modes'] = ['train']
input_fn_dict['classes'] = ['cats', 'dogs']
input_fn_dict['nb_batches'] = 16
input_fn_dict['nb_buffers'] = 256
input_fn_dict['nb_repeats'] = 50
input_fn_dict['nb_rows'] = nb_rows
input_fn_dict['nb_cols'] = nb_cols
input_fn_dict['nb_channels'] = nb_channels
input_fn_dict['one_hot'] = True
input_fn_dict['input_name'] = 'input_1'
input_fn_dict['eval_data_dir'] = '/mnt/data/Personal/coursera/DL/data/dogscats/sample'
eval_fn_dict['data_dir'] = input_fn_dict

# Log level
tf.logging.set_verbosity(tf.logging.DEBUG)

def main():

    # 1. define the model
    model = lenet_tf_keras.model(
            nb_classes=len(input_fn_dict['classes']),
            nb_rows=nb_rows, nb_cols=nb_cols, nb_channels=nb_channels)

    model.summary()

    input_fn_dict['input_name'] = model.input_names[0]

    # 2. run the estimator
    model_est = est_keras.run_from_generator(
            model, input_func=dg.get_img_generator, input_func_dict=input_fn_dict,
            eval_fn_dict=eval_fn_dict, nb_epochs=10, optimizer=None, model_dir='lenet')



if __name__ == '__main__':
    main()


#model.fit_generator(
#        train_generator,
#        steps_per_epoch=nb_train_samples // batch_size,
#        epochs=epochs,
#        validation_data=validation_generator,
#        validation_steps=nb_validation_samples // batch_size)
