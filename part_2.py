import skimage.util
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters, segmentation, morphology, measure
from skimage.color import rgb2gray

import skimage as sk
if __name__ == '__main__':
    picture = np.array(Image.open("scissors_col_2.jpg"))

    grayscale = rgb2gray(picture)

    grayscale = sk.util.random_noise(grayscale, 'gaussian', mean=0.0, var=0.01)

    thresh = sk.filters.threshold_otsu(grayscale)

    bw = morphology.closing(grayscale > thresh, morphology.square(5))

    cleared = segmentation.clear_border(bw)

    label = sk.measure.label(cleared)

    applied_props = sk.measure.regionprops(label)
    l2rgb = sk.color.label2rgb(label)

    for x in applied_props:
        applied_props.sort(key=lambda x: x.euler_number,reverse=False)
        if x.euler_number <= -1:
            applied_props.sort(key=lambda x: x.area, reverse=False)

    plt.imshow(applied_props[0].image)
    plt.title("Area = " + str(applied_props[0].area) + " Centroid = " + str(applied_props[0].centroid))
    plt.show()
