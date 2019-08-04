import numpy as np
import matplotlib.pyplot as plt
import cv2


def conv1D(inSignal, kernel1):
    length_inSignal = np.size(inSignal)
    length_kernel1 = np.size(kernel1)
    
    ans = np.zeros(length_inSignal + length_kernel1 -1)

    for m in np.arange(length_inSignal):
        for n in np.arange(length_kernel1):
            ans[m+n] = ans[m+n] + inSignal[m]*kernel1[n]
            
    return ans

def conv2D(inImage, kernel2):
    # image & kernel dimensions
    (iH, iW) = inImage.shape[:2]
    (kH, kW) = kernel2.shape[:2]
    
    # padding the borders of the input image so its size (width and height) are not reduced
    pad = (kW - 1) // 2
    inImage = cv2.copyMakeBorder(inImage, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = inImage[y - pad:y + pad + 1, x - pad:x + pad + 1]
            # performing the convolution
            k = (roi * kernel2).sum()
            # storing the convolved value in the output image
            output[y - pad, x - pad] = k
    
    # rescaling the output image to be [0, 255]
    norm_image = cv2.normalize(output, None, alpha=0, beta=255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image = np.uint8(norm_image)
    
    return norm_image

def convDerivative(inImage):
    # inImage dimensions
    height = inImage.shape[0]
    width = inImage.shape[1]
    
    # kernel to derive rows
    kerX = np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]])
    # kernel to derive columns
    kerY = np.array([[1, 1, 1],
                     [0, 0, 0],
                     [-1, -1, -1]])
    
    imgX = np.zeros((height, width), dtype="float32")
    imgY = np.zeros((height, width), dtype="float32")
    
    imgX = conv2D(inImage, kerX)
    imgY = conv2D(inImage, kerY)
    
    # Creating a new image & calculating magnitude(sqrt(imgX**2 + imgY**2))
    imgMagnitude = np.zeros((inImage.shape[0], inImage.shape[1]), dtype="float32")
   
    # normalizing values to [-255,255] in order to make the final image areas with no edges to be black
    #                                                                   and areas with edges to be colored white.
    imgX = cv2.normalize(imgX, None, alpha=-255, beta=255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    imgY = cv2.normalize(imgX, None, alpha=-255, beta=255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    imgX = imgX**2
    imgY = imgY**2

    for i in range(imgMagnitude.shape[0]):
        for j in range(imgMagnitude.shape[1]):
            imgMagnitude[i,j] = math.sqrt(imgX[i,j] + imgY[i,j])

    return imgMagnitude

# making a Gaussian kernel by a given size and sigma (by defualt is 1)    
def gaussKernel(kernelSize, sigma=1):
    m, n = [(ss-1.)/2. for ss in kernelSize]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h   

def blurImage1(inImage, kernelSize):
    if(kernelSize[0] != kernelSize[1]):
        print("kernelSize needs to be the squared")

    elif(kernelSize[0] % 2 == 0 and kernelSize[1] % 2 == 0):
        print("kernelSize is even, you should choose an odd number")

    else:
        gaussianK = gaussKernel(kernelSize, 1)
        blurredImage = conv2D(inImage, gaussianK)
        
        # Plotting the original image and blurred image
        plt.subplot(122),plt.imshow(inImage, cmap = 'gray'),plt.title('Original')
        plt.xticks([]), plt.yticks([])

        plt.subplot(121),plt.imshow(blurImage, cmap = 'gray'),plt.title('Gaussian Blur')
        plt.xticks([]), plt.yticks([])
        plt.show()
        
        return blurredImage
        
def blurImage2(inImage, kernelSize):
    if(kernelSize[0] != kernelSize[1]):
        print("kernelSize needs to be the squared")
        
    elif(kernelSize[0] % 2 == 0 and kernelSize[1] % 2 == 0):
        print("kernelSize is even, you should choose an odd number")

    else:
        gaussianK = cv2.getGaussianKernel(kernelSize[0],10)
        gray = cv2.filter2D(inImage,-1,gaussianK)

        plt.subplot(122),plt.imshow(inImage, cmap = 'gray'),plt.title('Original')
        plt.xticks([]), plt.yticks([])

        plt.subplot(121),plt.imshow(gray, cmap = 'gray'),plt.title('Gaussian Blur')
        plt.xticks([]), plt.yticks([])

        plt.show()
        return gray

