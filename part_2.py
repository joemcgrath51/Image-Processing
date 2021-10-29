import skimage.util
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, transform, filters, util, restoration, segmentation, morphology, metrics
from skimage.color import rgb2gray

import skimage as sk
if __name__ == '__main__':
    picture = np.array(Image.open("/home/joe/PycharmProjects/Image-Processing/scissors_col_2.jpg"))

    fig1, axes1 = plt.subplots(7, 4, figsize=(30, 20))
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

    #ax[14].imshow(l2rgb)
    # ax[i].imshow(cleared, cmap=plt.cm.gray)
    # ax[i].set_title("Cleared")

    # i = i + 1

    b = []
    e_1 = rp[1].euler_number
    e_2 = rp[9].euler_number
    t = []

    for x in range(len(rp)):
        print("AREA = " + str(rp[x].area))

        print("CENTROID = " + str(rp[x].centroid))

        print("Euler Number = " + str(rp[x].euler_number))

        print("Solidity = " + str(rp[x].solidity))

        print("Perimeter = " + str(rp[x].perimeter))

        t.append(transform.resize(rp[x].image, [255,255]))
        b.append(skimage.img_as_float(t[x]))

        #ax[x].imshow(rp[x], cmap=plt.cm.gray)
        #ax[x].imshow(b[x])

        #print(b[0])
    images = []
    count = 0
    for x in range(len(b)):
        for y in range(x+1,len(b)):
            mse = skimage.metrics.mean_squared_error(b[x], b[y])
            #if rp[x].euler_number == rp[y].euler_number: # and (0.2 < mse < 0.25):

            ssim = metrics.structural_similarity(t[x], t[y], data_range=t[y].max() - t[y].min())
            # print(rp[x].euler_number)
            # print(rp[y].euler_number)
            # count = count + 1
            # print(count)
            if ssim > 0.5 and (rp[x].area/rp[y].area) < 0.3 :
                #print("SSIM = ", ssim)
                print("AREA = ", rp[x].area/rp[y].area)
                images.append(t[x])
                images.append(t[y])
    print(len(images))
    for x in range(len(images)):
        ax[x].imshow(images[x])



    # ax[i].set_title("Labels")

    fig1.tight_layout()

    plt.show()
