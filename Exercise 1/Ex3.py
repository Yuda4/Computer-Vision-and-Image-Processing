import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def GaussianPyramid(im, maxLevels):
    layer = im.copy()
    gaussian_pyramid = [layer]

    for i in range(maxLevels):
        layer = cv2.pyrDown(layer)
        gaussian_pyramid.append(layer)
        
    return gaussian_pyramid

def LaplacianPyramid(im, maxLevels):
    
    gaussian_pyramid = GaussianPyramid(im, maxLevels)
    layer = gaussian_pyramid[maxLevels]
    laplacian_pyramid = [layer]
    for i in range(maxLevels, 0, -1):
        size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
        laplacian_pyramid.append(laplacian)

    return laplacian_pyramid
    
def LaplacianToImage(lpyr):
    reconstructed_image = lpyr[0]
    images = []
    hight = []
    widthTotal = 0

    for i in range(1, len(lpyr)):
        size = (lpyr[i].shape[1], lpyr[i].shape[0])
        reconstructed_image = cv2.pyrUp(reconstructed_image, dstsize=size)
        reconstructed_image = cv2.add(reconstructed_image, lpyr[i])
        images.append(reconstructed_image)
        widthTotal += reconstructed_image.shape[1]
        hight.append(size[0])

    lengthI = len(images)
    testin = []

    for i in range(lengthI):
        img = Image.fromarray(images[i])
        img_w, img_h = img.size
        background = Image.new('RGBA', (img_w, hight[lengthI-1]), (255, 255, 255, 255))
        bg_w, bg_h = background.size
        offset = (0, 0)
        background.paste(img, offset)
        testin.append(background)
        background.save('test'+str(i)+'.png')
    
    if(lengthI == 4):
        numpy_horizontal = np.hstack((testin[len(testin)-1], testin[len(testin)-2],testin[len(testin)-3], testin[len(testin)-4]))
    elif(lengthI == 3):
        numpy_horizontal = np.hstack((testin[len(testin)-1], testin[len(testin)-2],testin[len(testin)-3]))
    elif(lengthI == 2):
        numpy_horizontal = np.hstack((testin[len(testin)-1], testin[len(testin)-2]))
    imgFinal = Image.fromarray(numpy_horizontal)
    imgFinal.save('final.png')
    cv2.imshow('Numpy Horizontal', numpy_horizontal)
    cv2.waitKey()

    return reconstructed_image
    

