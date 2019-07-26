import numpy as np
import matplotlib.pyplot as plt
import cv2

# imReadAndConvert function with 2 variables:
#       filename - string containing the image filename to read.
#       representation - representation code, defining if the output should be either a:
#                       (1) grayscale image  
#                       (2) or an RGB image
def imReadAndConvert(filename, representation):
    # Loading an image
    img = cv2.imread(filename)
    # normolaize pixels
    img = img*(1./255)
    if representation == 1:
        # representation = 1 means grayscale image
        img = RGBtoGray(img)
    else:
        # representation = 2 means RGB image
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        
    return img

# Calculate the grayscale values so as to have the same luminance as the original color image
# RGB color model to a grayscale representation of its luminance
def RGBtoGray(img):
    
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])
    
    R = (R *.299)
    G = (G *.587)
    B = (B *.114)
    
    for i in range(3):
        img[:,:,i] = (R+G+B)
    
    return img

# Displaying the image, depending on representation paramter if to change it to grayscale or not
def imDisplay(filename, representation):
    #img = imReadAndConvert(filename, representation)
    
    img2 = cv2.imread(filename)
    img2 = histogramEqualize(img2)
    #imgYIQ = transformRGB2YIQ(img2)
    
    #imgRGB = transformYIQ2RGB(img2)
    #cv2.imwrite('222.jpg',imgRGB) # saving image
    #plt.imshow(imgRGB)
    
    #plt.imshow(imgYIQ)
    plt.imshow(img2)
    plt.show()
    
# transforming a RGB values to YIQ values    
def transformRGB2YIQ(img):
    
    # taking sizes of input to make a new image
    height = img.shape[0]
    width = img.shape[1]
    
    # creating a new matrix, same size as input with 3 dimension
    YIQ = np.zeros((height,width,3))
    # splitting each dimension to a matrix, it is BGR in opencv functions
    b,g,r = cv2.split(img)
    
    imgY = 0.299 * r + 0.587 * g + 0.114 * b
    imgI = 0.596 * r - 0.275 * g - 0.321 * b
    imgQ = 0.212 * r - 0.523 * g + 0.311 * b
    
    YIQ = cv2.merge([imgY,imgI,imgQ])
    #saving an image as YIQ
    #cv2.imwrite('yiq_photo.jpg',cv2.merge([imgQ,imgI,imgY]))
    YIQ = YIQ*(1./255)
    return(YIQ)

# transforming a YIQ values to RGB values
def transformYIQ2RGB(img):
    print(img.shape)
    # taking sizes of input to make a new image
    height = img.shape[0]
    width = img.shape[1]
    
    # creating a new matrix, same size as input with 3 dimension
    RGB = np.zeros((height,width,3))
    q,i,y = cv2.split(img)
    img = cv2.merge([y,i,q])

    imgR = 1 * y + 0.956 * i + 0.619 * q
    imgG = 1 * y - 0.272 * i - 0.647 * q
    imgB = 1 * y - 1.106 * i + 1.703 * q
    
    RGB = cv2.merge([imgR,imgG,imgB])
    
    RGB = RGB*(1./255)
    print(RGB)
    # saving an image as RGB
    #cv2.imwrite('rgb_photo.jpg',cv2.merge([imgB,imgG,imgR]))
    #print(RGB)
    return(RGB)

#
def histogramEqualize(imOrig):
    
    
    B,G,R = cv2.split(imOrig)

    if(np.array_equal(R, G) and np.array_equal(G, B)):

        print("Grayscale image")
    else:
        
        imOrig = cv2.merge([R,G,B])
        imOrig = RGBtoGray(imOrig)
        
        hist = cv2.calcHist([imOrig],[0],None,[256],[0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max()/ cdf.max()

        plt.plot(cdf_normalized, color = 'r')
        original = plt.hist(imOrig.flatten(),256,[0,256], color = 'b')
        plt.xlim([0,256])
        plt.legend(('cdf','histogram'), loc = 'upper left')
        plt.show()

        equ = cv2.equalizeHist(imOrig[:,:,0])

        hist2 = cv2.calcHist([equ],[0],None,[256],[0,256])
        cdf2 = hist.cumsum()
        cdf2_normalized = cdf2 * hist2.max()/ cdf2.max()

        plt.plot(cdf2_normalized, color = 'r')
        equalize = plt.hist(equ.flatten(),256,[0,256], color = 'b')
        plt.xlim([0,256])
        plt.legend(('cdf','histogram'), loc = 'upper left')
        plt.show()
        
        cv2.waitKey(0)


        res = np.hstack((imOrig[:,:,1], equ)) #stacking images side-by-side
        cv2.imwrite('equ2.jpg',res)

        #plt.hist(imOrig.ravel(),256,[0,256]) 
        #plt.show()
    return imOrig

def hisAlgo():

    img = cv2.imread('grayphoto.jpg')
    
    cv2.imshow('gray',img)
    # calcHist(choosen img,grayscale chanle,mask image, # of bin, ranges)
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    plt.plot(cdf_normalized, color = 'r')
    plt.hist(img.ravel(),256,[0,256])
    plt.title('Histogram for gray scale picture')
    
    plt.show()
    cv2.waitKey(0)


    

im = imDisplay(filename = 'apple.jpg', representation = 2)
#hisAlgo()


