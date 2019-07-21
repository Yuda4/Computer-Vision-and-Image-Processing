import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import cv2

# imReadAndConvert function with 2 variables:
#       filename - string containing the image filename to read.
#       representation - representation code, defining if the output should be either a:
#                       (1) grayscale image  
#                       (2) or an RGB image
def imReadAndConvert(filename, representation):
    # Load an image and coloring depeding on representation code 1 or 2
    img = cv2.imread(filename)
    
    if representation == 1:
        # representation = 1 means the user want the photo to be grayscale
        # This function first normalize each pixel and then converts RGB photo to Gray
        img = img*(1./255)
        img = RGBtoGray(img)
    else:
        # This means representation is 2 and the user wants RGB photo, so we normalizing its pixels
        img = img*(1./255)
        b,g,r = cv2.split(img)
        img2 = cv2.merge([r,g,b])
        return img2
    
    #print(img)
    #print(type(img))
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

# Displaying the image, depends on representation paramter if to change it to grayscale or not
def imDisplay(filename, representation):
    #img = imReadAndConvert(filename, representation)
    
    
    img2 = cv2.imread(filename)
    #imgYIQ = transformRGB2YIQ(img2)
    
    imgRGB = transformYIQ2RGB(img2)
    #cv2.imwrite('222.jpg',imgRGB) # saving image
    plt.imshow(imgRGB)
    #plt.imshow(imgYIQ)
    plt.show()
    #imgYIQ = imgYIQ*255
    
def transformRGB2YIQ(img):

    height = img.shape[0]
    width = img.shape[1]
    
    YIQ = np.zeros((height,width,3))
    b,g,r = cv2.split(img)
    #img2 = cv2.merge([r,g,b])
    imgY = 0.299 * r + 0.587 * g + 0.114 * b
    imgI = 0.596 * r - 0.275 * g - 0.321 * b
    imgQ = 0.212 * r - 0.523 * g + 0.311 * b
    
    YIQ = cv2.merge([imgY,imgI,imgQ])
    #cv2.imwrite("onePic123.jpg", YIQ)
    YIQ = YIQ*(1./255)
    print(YIQ)
    return(YIQ)

def transformYIQ2RGB(img):
    
    height = img.shape[0]
    width = img.shape[1]
    
    RGB = np.zeros((height,width,3))
    y,i,q = cv2.split(img)
    img2 = cv2.merge([y,i,q])
    imgR = 1 * y + 0.956 * i + 0.619 * q
    imgG = 1 * y - 0.272 * i - 0.647 * q
    imgB = 1 * y - 1.106 * i + 1.703 * q
    
    RGB = cv2.merge([imgR,imgG,imgB])
    RGB = RGB*(1./255)
    #RGB[:,:,1] = RGB[:,:,1] * (-1)
    #RGB[:,:,2] = RGB[:,:,2] / 2.
    print(RGB)
    return(RGB)





im = imDisplay(filename = 'onePic2.jpg', representation = 2)

# 3->2 
# hist = np.histogram(image.flatten(),256,[0,256])[0]

#cv2.destroyAllWindows() # closing all images


