import numpy as np
import numpy.matlib
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
    if img is not None:
        # normolaize pixels
        img = img*(1./255)
        if representation == 1:
            # representation = 1 means grayscale image
            img = RGBtoGray(img)
        else:
            # representation = 2 means RGB image
            b,g,r = cv2.split(img)
            img = cv2.merge([r,g,b])
    else:
        print("Invalid path for image")
        
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
    img = imReadAndConvert(filename, representation)
    plt.imshow(img)
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
    YIQ = YIQ*(1./255)
    
    return(YIQ)

# transforming a YIQ values to RGB values
def transformYIQ2RGB(img):
    
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
    return(RGB)

# Equalize an input image, if input is an RGB image it equalize it with Y channle of YIQ values, 
#                             otherwise input image is grayscale and do it on gray values
def histogramEqualize(imOrig):
    
    B,G,R = cv2.split(imOrig)

    # Grayscale image
    if(np.array_equal(R, G) and np.array_equal(G, B)):      
        imEq,histOrig,histEq = histogram(imOrig)

    else:
        imOrig = cv2.merge([B,G,R])
        YIQimage = transformRGB2YIQ(imOrig)

        YIQimage = YIQimage.astype(np.float32)

        Q,I,Y = cv2.split(YIQimage)
        YIQimage = cv2.merge([Q,I,Y])
        Y = Y*255
        imEq,histOrig,histEq = histogram(Y)

    imOrig = cv2.merge([R,G,B])
    showOutput(imOrig, imEq, histOrig, histEq)

    return imEq, histOrig, histEq

def histogram(img):

    flat = img.flatten()
    flat = flat.astype(int)
    
    histOrig = cv2.calcHist([img],[0],None,[256],[0,256])
    cdf = histOrig.cumsum()

    nj = (cdf - cdf.min()) * 255
    N = cdf.max() - cdf.min()
    cdf = nj / N
    cdf = cdf.astype('uint8')
    
    # get the value from cumulative sum for every index in flat, and set that as newImage
    imEq = cdf[flat]

    histEq = cv2.calcHist([imEq],[0],None,[256],[0,256])
    imEq = np.reshape(imEq, img.shape)

    return imEq, histOrig, histEq

def showOutput(imOrig, imEq, histOrig, histEq):
    # set up side-by-side image display
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(10)

    # display the original image
    fig.add_subplot(1,2,1)
    plt.imshow(imOrig)

    # display the equalized image
    fig.add_subplot(1,2,2)
    plt.imshow(imEq)
    fig.suptitle('Left is original photo, Right is equalized photo', fontsize=16)
    plt.show(block=True)

    # set up side-by-side image display
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(10)

    # display the original histogram image
    fig.add_subplot(1,2,1)
    plt.plot(histOrig)

    # display the equalized histogram image
    fig.add_subplot(1,2,2)
    plt.plot(histEq)
    fig.suptitle('Left is histogram of original photo, Right is histogram of equalized photo', fontsize=12)
    plt.show(block=True)

#imOrig - input grayscale or RGB image to be quantized.
#nQuant - number of intensities your output imQuant image should have.
#nIter  - maximum number of iterations of the optimization procedure.
def quantizeImage(imOrig, nQuant, nIter ):
    #imOrig = cv2.imread('applegray.jpg', cv2.IMREAD_COLOR)
    plt.imshow(cv2.cvtColor(imOrig, cv2.COLOR_BGR2RGB))
    plt.show()

    img_data = imOrig / 255.0
    img_data = img_data.reshape((-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    img_data = img_data.astype(np.float32)
    compactness, labels, centers = cv2.kmeans(img_data, nQuant, None, criteria, nIter,  cv2.KMEANS_RANDOM_CENTERS)
    
    new_colors = centers[labels].reshape((-1, 3))
    img_recolored = new_colors.reshape(imOrig.shape)
    plt.imshow(cv2.cvtColor(img_recolored, cv2.COLOR_BGR2RGB))
    plt.title('16-color image')
    plt.show()

    return labels, centers


def main():
    im = imDisplay(filename = 'apple.jpg', representation = 2)
    

if __name__ == '__main__':
    main()
