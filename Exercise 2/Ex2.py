import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


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
def gaussianKernel(kernelSize, sigma=1):
    kernelSize = int(kernelSize[0]) // 2
    x, y = np.mgrid[-kernelSize:kernelSize+1, -kernelSize:kernelSize+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g 

def blurImage1(inImage, kernelSize):
    if(kernelSize[0] != kernelSize[1]):
        print("kernelSize needs to be the squared")

    elif(kernelSize[0] % 2 == 0 and kernelSize[1] % 2 == 0):
        print("kernelSize is even, you should choose an odd number")

    else:
        gaussianK = gaussianKernel(kernelSize, 1)
        blurredImage = conv2D(inImage, gaussianK)
        
        # Plotting the original image and blurred image
        plt.subplot(122),plt.imshow(inImage, cmap = 'gray'),plt.title('Original')
        plt.xticks([]), plt.yticks([])

        plt.subplot(121),plt.imshow(blurredImage, cmap = 'gray'),plt.title('Gaussian Blur')
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

def edgeDetectionSobel(I):

    # Smoothing
    gaussianK = gaussianKernel((3,3), 1)
    I = conv2D(I, gaussianK)

    # X & Y Kernels
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = conv2D(I, Kx)
    Iy = conv2D(I, Ky)

    plt.subplot(121),plt.imshow(Ix, cmap = 'gray'),plt.title('Ix')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(Iy, cmap = 'gray'),plt.title('Iy')
    plt.xticks([]), plt.yticks([])
    plt.show()

    edgeImage1 = calcMagnitude(Ix, Iy, I.shape[0], I.shape[1])

    img = cv2.GaussianBlur(I,(3,3),0)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    edgeImage2 = np.hypot(sobelx, sobely)

    plt.subplot(121),plt.imshow(edgeImage1, cmap = 'gray'),plt.title('Manual Sobel')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edgeImage2, cmap = 'gray'),plt.title('openCV2 Sobel')
    plt.xticks([]), plt.yticks([])
    plt.show()

    return edgeImage1, edgeImage2

# only laplacian, without smoothing
def edgeDetectionZeroCrossingSimple(I):
    laplacianMat = np.array([[1, 1, 1],
                             [1, -8, 1],
                             [1, 1, 1]])

    edgeImage1 = conv2D(I, laplacianMat)
    plt.imshow(edgeImage1, cmap='gray')
    plt.show()
    
    cv2Lap = cv2.Laplacian(I, cv2.CV_16S, 3)
    edgeImage2 = cv2.convertScaleAbs(cv2Lap)

    plt.subplot(121),plt.imshow(edgeImage2, cmap = 'gray'),plt.title('Manual laplacian')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edgeImage2, cmap = 'gray'),plt.title('Auto laplacian')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    
    return edgeImage1, edgeImage2

def edgeDetectionZeroCrossingLOG(image):
    gaussian_mask=np.array([[(0.109),(0.111),(0.109)],[(0.111),(0.135),(0.111)],[(0.109),(0.111),(0.109)]],dtype='f')
    lap_mask = np.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],[0,1,2,1,0],[0,0,1,0,0]])

    sobel_mask_h = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    sobel_mask_v = np.array([[3,2,1,0,-1,-2,-3],[4,3,2,0,-2,-3,-4],[5,4,3,0,-3,-4,-5],[6,5,4,0,-4,-5,-6],[5,4,3,0,-3,-4,-5],[4,3,2,0,-2,-3,-4],[3,2,1,0,-1,-2,-3]])

    sobel_mask_1 = np.array([[2,1,0],[1,0,-1],[0,-1,-2]])
    sobel_mask_2 = np.array([[0,-1,-2],[1,0,-1],[2,1,0]])
    sobel_mask_3 = np.array([[-2,-1,0],[-1,0,1],[0,1,2]])
    sobel_mask_4 = np.array([[0,1,2],[-1,0,-1],[-2,-1,0]])


    logImg = cv2.filter2D(image,-1,gaussian_mask)

    logDst = cv2.filter2D(image,-1,lap_mask)

    #marking the zero crossing edges from top to bottom & left to Right
    (r,c) = logDst.shape
    array2 = logDst
    zc2 = np.zeros_like(array2)
    for x in range(r):
        for y in range (c):
            zc2[x,y]=255

    for x in range(r):
        temp=np.zeros(c)
        for y in range (c):
            temp[y]=array2[x,y]
        zero_crossings2 = np.where(np.diff(np.sign(temp)))[0]
        for q in range (len(zero_crossings2)):
            zc2[x,zero_crossings2[q]]=array2[x,zero_crossings2[q]]

    for x in range(c):
        temp=np.zeros(c)
        for y in range (r):
            temp[y]=array2[y,x]
        zero_crossings2 = np.where(np.diff(np.sign(temp)))[0]
        for q in range (len(zero_crossings2)):
            zc2[zero_crossings2[q],x]=array2[zero_crossings2[q],x]


    edgeImage1 = zc2	
    cv2.imshow("LOG ZeroCrossing.jpg",zc2)
    cv2.imwrite("LOG ZeroCrossing.jpg",zc2)

    #Removing weak edges using sobel filter. We use sobel in all 6 directions
    dst_h = cv2.filter2D(logImg,-1,sobel_mask_h)
    dst_v = cv2.filter2D(logImg,-1,sobel_mask_v)
    dst_1 = cv2.filter2D(logImg,-1,sobel_mask_1)
    dst_2 = cv2.filter2D(logImg,-1,sobel_mask_2)
    dst_3 = cv2.filter2D(logImg,-1,sobel_mask_3)
    dst_4 = cv2.filter2D(logImg,-1,sobel_mask_4)
    
    #removing weak edges that do not have first derivative(in sobel magnitude image) from zero crossing image zc2
    for x in range(r):
        for y in range (c):
            if(dst_h[x,y]<30 and dst_v[x,y]<30 and (dst_1[x,y]<30)and(dst_2[x,y]<30)and(dst_3[x,y]<30)and(dst_4[x,y]<30) and zc2[x,y]<100):
                edgeImage1[x,y]=255
            if(dst_h[x,y]>200 or dst_v[x,y]>200 or(dst_1[x,y]>200)or(dst_2[x,y]>200)or(dst_3[x,y]>200)or(dst_4[x,y]>200) and zc2[x,y]<60):
                edgeImage1[x,y]=zc2[x,y]


    cv2.imshow("Final LOG.jpg",edgeImage1)
    cv2.imwrite("LoG_Cat.jpg",edgeImage1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def edgeDetectionCanny(I):

    edges = cv2.Canny(I,100,200)

    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()

def houghCircle(I,minRadius,maxRadius):
    img = cv2.imread('opencv_logo.png',0)
    img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
     
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius)
    
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    
    cv2.imshow('detected circles',cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
