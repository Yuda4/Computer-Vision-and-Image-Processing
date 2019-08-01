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
    (iH, iW) = inImage.shape[:2]
    (kH, kW) = kernel2.shape[:2]
 
    pad = (kW - 1) // 2
    inImage = cv2.copyMakeBorder(inImage, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = inImage[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * kernel2).sum()
            output[y - pad, x - pad] = k

    norm_image = cv2.normalize(output, None, alpha=0, beta=255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image = np.uint8(norm_image)
    
    return norm_image

