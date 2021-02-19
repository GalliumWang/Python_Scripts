import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,exp

def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def idealFilterLP(imgShape,D0=10):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base

def idealFilterHP(imgShape,D0=10):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 0
    return base


def butterworthLP(imgShape,D0=10,n=10):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

def butterworthHP(imgShape,D0=10,n=10):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1-1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

def gaussianLP(imgShape,D0=10):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def gaussianHP(imgShape,D0=10):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base


def customLP():
    pass

def customHP():
    pass

filter_type_dict={"idealLP":idealFilterLP,
           "idealHP":idealFilterHP,
           "butterLP":butterworthLP,
           "butterHP":butterworthHP,
           "guassLP":gaussianLP,
           "guassHP":gaussianHP
          }

def Img_Process(filename,filter_type):
    
    # TODO
    plt.figure(figsize=(16, 9), constrained_layout=False)

    img = cv2.imread(filename, 0)
    # plt.subplot(161), plt.imshow(img, "gray"), plt.title("Original Image")
    original = np.fft.fft2(img)
    # plt.subplot(162), plt.imshow(np.log(1+np.abs(original)), "gray"), plt.title("Spectrum")
    center = np.fft.fftshift(original)
    # plt.subplot(163), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Centered Spectrum")
    PassCenter = center * filter_type_dict[filter_type](img.shape)
    # plt.subplot(164), plt.imshow(np.log(1+np.abs(PassCenter)), "gray"), plt.title("Centered Spectrum multiply Low Pass Filter")
    Pass = np.fft.ifftshift(PassCenter)
    # plt.subplot(165), plt.imshow(np.log(1+np.absPass)), "gray"), plt.title("Decentralize")
    inverse_Pass = np.fft.ifft2(Pass)
    # plt.subplot(166), plt.imshow(np.abs(inverse_Pass), "gray"), plt.title("Processed Image")

    return inverse_Pass




def process_factory(filename,filter_type,showResult=True,saveResult=False):
    # try:
    #     Img_Process(filename,filter_type)
    # except:
    #     raise Exception("Unkown Error in process")

    # include complex number
    processed_img=Img_Process(filename,filter_type)

    if(showResult):
        plt.imshow(np.abs(processed_img), "gray")
        plt.show()

    if(saveResult):
        cv2.imwrite(filename[:-4]+"_{}.jpg".format(filter_type),np.abs(processed_img))



process_factory("example.jpg","guassLP")

