import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_weights(layer_weight, img_shape, idx=0):


    nb_imgs = layer_weight.shape[1]

    # 1. we want (rows, 5 cols)
    nb_rows = int(nb_imgs / 5)
    print("nb_imgs = ", nb_imgs, " , nb_rows = ", nb_rows)

    # 2. get min and max of weights
    w_min = np.min(layer_weight)
    w_max = np.max(layer_weight)

    # 3. create figure axes
    fig, axes = plt.subplots(nb_rows, 5)
    # adjust subplots
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i,ax in enumerate(axes.flat):

        if i < nb_imgs:
            # Reshape the weights
            img = layer_weight[:,i].reshape(img_shape)

            # set label and show the image
            ax.set_xlabel("Weights : {0}".format(i))
            ax.imshow(img, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

    # Show the plot
    plt.draw()
    plt.savefig('test'+str(idx)+'.png')
