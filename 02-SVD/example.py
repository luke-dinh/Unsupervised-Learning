import numpy as np
import matplotlib.pyplot as plt
import imageio

#a. read the image
img1 = imageio.imread("02-SVD/image1.jpg")

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

# plt.subplot(1,2,1)
# plt.title("Original Image")
# plt.imshow(new_img, cmap="gray")

# plt.subplot(1,2,2)
# plt.title("SVD Image")
# plt.imshow(imgSVD, cmap="gray")

# plt.show()

# b. SVD with rank 1
rank_1_image = None
row = int(new_img.shape[0])
col = int(new_img.shape[1])

u_1 = np.reshape(U[:,0], (row, 1))
s_1 = S[0,0]
v_1 = np.reshape(V[0,:], (1, col))

rank_1_image = np.dot(U, S*V)

# plt.title("Rank 1 Image")
# plt.imshow(rank_1_image, cmap="gray")
# plt.show()

# c. Best rank:
rank20image = None
rank20image = np.zeros(np.shape(new_img))

for i in range(20):
    u20 = np.reshape(U[:,i], (row, 1))
    s20 = S[i,i]
    v20 = np.reshape(V[i,:], (1, col))
    rank20image += np.dot(u20, s20*v20)

plt.title("Rank 20 Image")
plt.imshow(rank20image, cmap="gray")
plt.show()