import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_layer_tensor(layer_tensor):


    # Iterate over all the images in the batch
    nb_imgs, nb_channels = layer_tensor.shape[0], layer_tensor.shape[3]

    # threshold nb_imgs
    if nb_imgs > 10:
        nb_imgs = 10

    # 1. find rows, cols based on nb_channels
    nb_rows = int(np.sqrt(nb_channels))
    nb_cols = int(nb_channels/nb_rows)

    # add figure and axes
    fig, axes = plt.subplots(nb_rows, nb_cols)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for j in range(nb_imgs):

        # 2. add subplots
        for i,ax in enumerate(axes.flat):

            # Reshape the weights
            img = np.squeeze(layer_tensor[int(j),:,:,int(i)])

            # 3. set label and show the image
            ax.set_xlabel("Batch, Tensor : {0},{1}".format(j,i))
            # get min and max of weights
            w_min = np.min(layer_tensor)
            w_max = np.max(layer_tensor)
            ax.imshow(img, vmin=w_min, vmax=w_max, cmap='seismic')

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

        # Show the plot
        plt.pause(0.001)
        #plt.savefig('test'+str(idx)+'.png')

    plt.close(fig)
