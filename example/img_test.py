from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Log level
tf.logging.set_verbosity(tf.logging.DEBUG)

NB_ROWS = 224
NB_COLS = 224
NB_CHANNELS = 3
sess = tf.Session()

def add_custom_layers(x):
    # read the image
    img_path = tf.read_file(tf.squeeze(tf.cast(x, tf.string)))
    img_u8 = tf.image.decode_jpeg(img_path, channels=3)
    # convert to float32
    img = tf.image.convert_image_dtype(img_u8, dtype=tf.float32)
    # resize
    img_cropped = tf.image.resize_image_with_crop_or_pad(img, NB_ROWS, NB_COLS)
    # normalize
    #img_cropped = (img_cropped - 128.)/128.
    #img_cropped = tf.expand_dims(img_cropped, 0)

    return img_cropped

def resize_images(x):
    return tf.map_fn(add_custom_layers, x, dtype=tf.float32)


# Our application logic will be added here
def main():
    
    x = tf.placeholder(tf.string, shape=(None, ))
    #y_ph = tf.placeholder(tf.float32, shape=(None,NB_CLASSES))
    
    # 1. build graph
    print('Creating graph...')
    img = resize_images(x)

    # 2. run the graph
    files = np.array(['/mnt/data/Personal/coursera/DL/data/dogscats/sample/train/cats/cat.10171.jpg', 
        '/mnt/data/Personal/coursera/DL/data/dogscats/sample/train/dogs/dog.10019.jpg'])

    #img_u8 = sess.run(img, feed_dict = {x:files})
    img_resized = sess.run(img, feed_dict = {x:files})
    print('resized : ', np.max(np.max(img_resized)), np.min(np.min(img_resized)))
    #print('original :', np.max(np.max(img_u8)), np.min(np.min(img_u8)))

    # print img types
    #print(img_u8.dtype, img_resized.dtype)
    print(img_resized.shape)

    for i in img_resized:
        # plot
        fig = plt.figure()
        #fig.add_subplot(1,2,1)
        #plt.imshow(img_u8)
        fig.add_subplot(1,2,2)
        plt.imshow(np.squeeze(i))
        plt.show()

    print('Closing the session')
    sess.close()

if __name__ == "__main__":
  #tf.app.run()
  main()
