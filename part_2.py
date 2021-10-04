from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, transform, filters, util, restoration, segmentation, morphology
from skimage.color import rgb2gray

import skimage as sk
if __name__ == '__main__':
    picture = np.array(Image.open("/home/joe/PycharmProjects/Image-Processing/scissors_col_2.jpg"))

    fig1, axes1 = plt.subplots(1, 3, figsize=(8, 4))
    grayscale = rgb2gray(picture)

    thresh = sk.filters.threshold_otsu(grayscale)
    bw = morphology.closing(grayscale > thresh, morphology.square(5))
    ax = axes1.ravel()

    i = 0
    ax[i].imshow(picture)
    ax[i].set_title("Original")

    i = i + 1
    ax[i].imshow(grayscale, cmap=plt.cm.gray)
    ax[i].set_title("Gray")

    i = i + 1

    cleared = segmentation.clear_border(bw)

    print(picture)
    ax[i].imshow(cleared, cmap=plt.cm.gray)
    ax[i].set_title("Cleared")

    fig1.tight_layout()

    plt.show()
