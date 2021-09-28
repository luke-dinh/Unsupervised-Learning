from skimage import data, img_as_ubyte, img_as_float
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

gray_images = {
    "cat": rgb2gray(img_as_float(data.chelsea())),
    "coffee": rgb2gray(img_as_float(data.coffee()))
}

# def compress_svd(image, k):

#     # Define U, S, V
#     U, S, V = np.linalg.svd(image, full_matrices=False)
#     comp_matrix = np.dot(U[:,:k], np.dot(np.diag(S[:k]), V[:k,:]))

#     return comp_matrix, k

def compress_image(img_name):

    img = gray_images[img_name]
    for i in range(5, 50, 5):
        U, S, V = np.linalg.svd(img, full_matrices=False)
        reconstimg = np.matrix(U[:, :i]) * np.diag(S[:i]) * np.matrix(V[:i, :])
        plt.imshow(reconstimg, cmap='gray')
        title = "n = %s" % i
        plt.title(title)
        plt.show()

compress_image(img_name="cat")