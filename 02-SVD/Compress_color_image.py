import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float

color_images = { 
    "cat": img_as_float(data.chelsea()),
    "astronaout": img_as_float(data.astronaut()),
    "coffee": img_as_float(data.coffee())
}

def compress_svd(image, k):
    U, S, V = np.linalg.svd(image, full_matrices=False)
    recon_matrix = np.dot(U[:,:k], np.dot(np.diag(S[:k]), V[:k,:]))

    return recon_matrix, k 

# Compress color image using SVD with reshaped image (From 3 dims H*W*C to 2 dims H*(W*C))
def compress_image(image_name):

    img = color_images[image_name]
    org_shape = img.shape
    img_reshape = img.reshape((org_shape[0], org_shape[1] * 3))
    for k in range(5, 50, 5):
        img_recon, _ = compress_svd(img_reshape, k)
        img_recon = img_recon.reshape(org_shape)
        compress_ratio = 100.0 * (k*(org_shape[0] + 
                                   org_shape[1]) +k)/(org_shape[0] * org_shape[1])
        
        plt.title("Compress Ratio: {:.2f}".format(compress_ratio) + "%")
        plt.imshow(img_recon)
        plt.show()

compress_image("coffee")