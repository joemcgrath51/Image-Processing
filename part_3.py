import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure, transform, filters, util, restoration, io
from skimage.color import rgb2gray
from skimage.morphology import disk, ball
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fig1, axes1 = plt.subplots(1, 3, figsize=(8, 4))
    #fig2, axes2 = plt.subplots(2, 3, figsize=(8, 4))

    ax = axes1.ravel()

    #ax2 = axes2.ravel()

    # Part A:

    original = io.imread("Selfie.png") # Acquire a colour image of your own face (front facing)
    colors = io.imread("reference.jpg")
    resized = transform.resize(original, [512, 512]) # resize it to 512x512 pixels
    colors = transform.resize(colors,[512,512])

    hist = exposure.match_histograms(resized, colors)

    i = 0
    ax[i].imshow(colors)
    ax[i].set_title("Colors")
    #
    ax[i+1].imshow(hist)
    ax[i+1].set_title("Resized")

    #
    ax[i+2].imshow(resized)
    ax[i+2].set_title("Gaussian noise")
    # #
    # ax[i+3].imshow(med_filter, cmap=plt.cm.gray)
    # ax[i+3].set_title("Medium Filter")
    #
    # ax[i+4].imshow(mean_filter, cmap=plt.cm.gray)
    # ax[i+4].set_title("Mean Filter")

    #fig1.tight_layout()
    #fig2.tight_layout()

    plt.show()
