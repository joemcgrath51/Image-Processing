import matplotlib.pyplot as plt
import numpy as np
from skimage import data, transform, filters, util, restoration, io
from skimage.color import rgb2gray
from skimage.morphology import disk, ball
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fig1, axes1 = plt.subplots(2, 3, figsize=(8, 4))
    fig2, axes2 = plt.subplots(2, 3, figsize=(8, 4))

    ax = axes1.ravel()
    ax2 = axes2.ravel()

    # Part A:

    original = io.imread("Selfie.png") # Acquire a colour image of your own face (front facing)
    resized = transform.resize(original, [512, 512]) # resize it to 512x512 pixels
    grayscale = rgb2gray(resized) # Convert this image to greyscale


    print("Max Value",grayscale.max())  # Find and display the minimum,maximum and mean greyscale values in this image
    print("Min Value",grayscale.min())
    print("Mean Value",grayscale.mean())

    # Part B

    # Taking the output image and apply Gaussian Noise mean 0.0 variance 0.1

    gaussian_img = util.random_noise(grayscale, 'gaussian', mean=0.0, var=0.01)

    # Part C
    # Apply 10x10 mean and median rank filter

    med_filter = filters.median(gaussian_img, disk(5))
    mean_filter = filters.rank.mean(gaussian_img, disk(5))

    # Part D
    for x in range(6):
        ax2[x].imshow(filters.gaussian(gaussian_img, x / 10), cmap=plt.cm.gray)
        ax2[x].set_title("Gaussian Filter at var = " + str(x/10))

        ax2[0].imshow(gaussian_img, cmap=plt.cm.gray)
        ax2[0].set_title("Gaussian noise")


    i = 0
    ax[i].imshow(original)
    ax[i].set_title("Original")
    #
    ax[i+1].imshow(grayscale, cmap=plt.cm.gray)
    ax[i+1].set_title("Grayscale")
    #
    ax[i+2].imshow(gaussian_img, cmap=plt.cm.gray)
    ax[i+2].set_title("Gaussian noise")
    #
    ax[i+3].imshow(med_filter, cmap=plt.cm.gray)
    ax[i+3].set_title("Medium Filter")

    ax[i+4].imshow(mean_filter, cmap=plt.cm.gray)
    ax[i+4].set_title("Mean Filter")

    fig1.tight_layout()
    fig2.tight_layout()

    plt.show()
