import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from skimage import io, feature
from scipy import ndimage

def disparitySSD(img_l, img_r, disp_range, k_size):
    
    stereo = cv2.StereoBM_create(disp_range, k_size)
    disparity = stereo.compute(img_l, img_r)
    
    disImage = Image.fromarray(disparity)
    disImage.save('disparity1.png')
    plt.imshow(disparity,'gray')
    plt.show()
    return disparity

def disparityNC(img_l, img_r, disp_range, k_size):
    # Load in both images, assumed to be RGBA 8bit per channel images

    left = np.asarray(img_l)
    right = np.asarray(img_r)    
    w, h = img_l.size  # assume that both images are same size   
    
    # Depth (or disparity) map
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w
       
    kernel_half = int(k_size / 2)    
    offset_adjust = 255 / disp_range  # this is used to map depth map output to 0-255 range
      
    for y in range(kernel_half, h - kernel_half):      
        print(".", end="", flush=True)  # let the user know that something is happening (slowly!)
        
        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_ssd = 65534
            
            for offset in range(disp_range):               
                ssd = 0
                ssd_temp = 0                            
                
                # v and u are the x,y of our local window search, used to ensure a good 
                # match- going by the squared differences of two pixels alone is insufficient, 
                # we want to go by the squared differences of the neighbouring pixels too
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        # iteratively sum the sum of squared differences value for this block
                        # left[] and right[] are arrays of uint8, so converting them to int saves
                        # potential overflow, and executes a lot faster 
                        ssd_temp = int(left[y+v, x+u]) - int(right[y+v, (x+u) - offset])  
                        ssd += ssd_temp * ssd_temp              
                
                # if this value is smaller than the previous ssd at this block
                # then it's theoretically a closer match. Store this value against
                # this block..
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset
                            
            # set depth output for this x,y location to the best match
            depth[y, x] = best_offset * offset_adjust
                                
    # Convert to PIL and save it
    Image.fromarray(depth).save('depth.png')
    return depth


def computeHomography(src_pnt, dst_pnt):
    matrixIndex = 0
    A = []
    for i in range(0, len(src_pnt)):
        x, y = src_pnt[i][0], src_pnt[i][1]
        u, v = dst_pnt[i][0], dst_pnt[i][1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    H = L.reshape(3, 3)

    M, mask = cv2.findHomography(src_pnt, dst_pnt, cv2.RANSAC)
    mask = mask.ravel()
    print(M)
    print(mask)

    return M, mask



if __name__ == '__main__':
    imgR = cv2.imread('pair0-R.png',0)
    imgL = cv2.imread('pair0-L.png',0)
    disparitySSD(imgL,imgR,16,15)

    img_l = Image.open("pair1-L.png").convert('L')
    img_r = Image.open("pair1-R.png").convert('L')
    disparityNC(img_l, img_r, 30, 6)

    src_pnt = np.array([[279, 552],
    [372, 559],
    [362, 472],
    [277, 469]])
    dst_pnt = np.array([[24, 566],
    [114, 552],
    [106, 474],
    [19, 481]])

    computeHomography(src_pnt, dst_pnt)


