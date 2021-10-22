import numpy as np
import matplotlib.pyplot as plt
import imageio

#a. read the image
img1 = imageio.imread("HW0/image1.jpg")

def rgb2gray(image):
    return 0.299*image[0] + 0.587*image[1] + 0.114*image[2]

gray_image = np.zeros((img1.shape[0], img1.shape[1]))

for r in range(len(img1)):
    for c in range(len(img1[r])):
        gray_image[r][c] = rgb2gray(img1[r][c])

def scaling(img):
    pixmax = np.max(img)
    pixmin = np.min(img)

    return (img - pixmax) / (pixmax - pixmin)

new_img = scaling(gray_image)

U, S, V = np.linalg.svd(new_img, full_matrices=False)
S = np.diag(S)
imgSVD = np.dot(U, np.dot(S, V))