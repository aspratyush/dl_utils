import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Activation, MaxPooling2D
from tensorflow.keras.layers import Dropout, Reshape, Flatten
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.utils import plot_model
from dl_utils.tf.plot_weights import plot_weights

# CUDA GPU
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'


def get_stats(X,Y):
    print("X : shape : (%d,%d)" % (X.shape), end='')
    print(",min : %f, max : %f" % (np.min(X), np.max(X)))
    print("Y : shape : (%d,%d)" % (Y.shape), end='')
    print(", min : %f, max : %f" % (np.min(Y), np.max(Y)))


def load_data(one_hot=False, nb_classes=10):
    from tensorflow.examples.tutorials.mnist import input_data

    # load data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=one_hot)
    x_train, y_train = mnist.train.images, mnist.train.labels
    x_test, y_test = mnist.test.images, mnist.test.labels
    x_validation, y_validation = mnist.validation.images, mnist.validation.labels

    if not(one_hot):
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=nb_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=nb_classes)
        y_validation = tf.keras.utils.to_categorical(y_validation, num_classes=nb_classes)

    # print stats
    print("train : ")
    get_stats(x_train, y_train)
    print("test : ")
    get_stats(x_test, y_test)
    print("validation : ")
    get_stats(x_validation, y_validation)

    return mnist, x_train, y_train, x_test, y_test, x_validation, y_validation


def build_model(use_softmax=False, nb_rows=28, nb_cols=28, nb_channels=1):
    model = Sequential()

    layers = [Reshape((nb_rows, nb_cols, nb_channels), input_shape=(1, 784)),
            Conv2D(16, (5, 5), activation='relu'),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            Conv2D(36, (5, 5), activation='relu'),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10)]

    # add layers to the model
    for layer in layers:
        model.add(layer)

    # softmax
    if use_softmax:
        model.add(Activation('softmax'))

    return model


def main():

    # 1. load data
    mnist, x_train, y_train, x_test, y_test, x_validation, y_validation = load_data()

    # 2. build model
    x = tf.placeholder(dtype=tf.float32, shape=(None,784))
    y = tf.placeholder(dtype=tf.float32, shape=(None,10))
    model = build_model()
    logits = model(x)
    model.summary()
    # plot the model
    plot_model(model)

    # 3. Loss and train_step
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
    loss = tf.reduce_sum(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    # 4. Accuracy
    y_pred = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    nb_epochs = 100
    nb_batches = 256
    # 5. run the training
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    acc_list = []
    avg_cost_list = []
    for epoch in range(nb_epochs):
        avg_cost = 0
        for i in range(int(x_train.shape[0]/nb_batches)):
            idx_start = i*nb_batches
            idx_end = (i+1)*nb_batches
            if (idx_end > x_train.shape[0]):
                idx_end = x_train.shape[0]

            # get batch
            x_batch, y_batch = x_train[idx_start:idx_end,:], y_train[idx_start:idx_end,:]
            _, c = sess.run([train_step, loss], feed_dict={x:x_batch, y:y_batch})

        avg_cost += c/nb_batches
        acc = sess.run(accuracy, feed_dict={x:x_validation, y:y_validation})
        avg_cost_list.append(avg_cost)
        acc_list.append(acc)
        if epoch % 10 == 0:
                        print("Epoch:", '%04d' % (epoch+1),
                    "Loss={:.9f}".format(avg_cost),
                    "Validation={:2.9f}".format(acc))

    print("Optimization finished...")

    # close the session
    sess.close()

    return avg_cost_list, acc_list

if __name__ == '__main__' :

    if (os.path.isfile('metrics.npy')):
        avg_cost, acc = np.load('metrics.npy')
    else:
        avg_cost, acc = main()
        np.save('metrics.npy', [avg_cost, acc], allow_pickle=True)

    # plot
    fig, ax = plt.subplots(2,1)
    fig.subplots_adjust(hspace=1.0, wspace=0.3)
    ax[0].plot(range(len(avg_cost)), avg_cost)
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('Loss')
    ax[0].grid(b=True)

    ax[1].plot(range(len(acc)), acc)
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('accuracy')
    ax[1].grid(b=True)
    plt.savefig('loss_accuracy.png')
    plt.show()
