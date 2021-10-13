from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, transform, filters, util, restoration, segmentation, morphology
from skimage.color import rgb2gray

import skimage as sk
if __name__ == '__main__':
    picture = np.array(Image.open("/home/joe/PycharmProjects/Image-Processing/scissors_col_2.jpg"))

    fig1, axes1 = plt.subplots(3, 5, figsize=(8, 4))
    grayscale = rgb2gray(picture)

    thresh = sk.filters.threshold_otsu(grayscale)
    bw = morphology.closing(grayscale > thresh, morphology.square(5))
    ax = axes1.ravel()

    # i = 0
    # ax[i].imshow(picture)
    # ax[i].set_title("Original")
    #
    # i = i + 1
    # ax[i].imshow(grayscale, cmap=plt.cm.gray)
    # ax[i].set_title("Gray")
    #
    # i = i + 1

    cleared = segmentation.clear_border(bw)

    label = sk.measure.label(cleared)

    rp = sk.measure.regionprops(label)
    l2rgb = sk.color.label2rgb(label)

    ax[14].imshow(l2rgb)
    # ax[i].imshow(cleared, cmap=plt.cm.gray)
    # ax[i].set_title("Cleared")

    # i = i + 1

    for x in range(len(rp)):
        print("AREA = " + str(rp[x].area))
        print("CENTROID = " + str(rp[x].centroid))

        ax[x].imshow(rp[x].image)

    # ax[i].imshow(rp[10].image)
    # ax[i].set_title("Labels")

    fig1.tight_layout()

    plt.show()
