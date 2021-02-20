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
    plt.figure(figsize=(16,9), constrained_layout=False)

    radius_paras=[30,60,90,120,150,180]
    # radius_paras=[30,100,150,200,250,300]
    # radius_paras=[300,350,400,450,500,550]
    subplot_pos=[231,232,233,234,235,236]

    for idx,radius in enumerate(radius_paras):
        print("iteration {}".format(idx))

        img = cv2.imread(filename, 0)
        original = np.fft.fft2(img)
        center = np.fft.fftshift(original)
        PassCenter = center * filter_type_dict[filter_type](img.shape,radius)
        Pass = np.fft.ifftshift(PassCenter)
        inverse_Pass = np.fft.ifft2(Pass)

        plt.subplot(subplot_pos[idx]), plt.imshow(np.abs(inverse_Pass), "gray"), plt.title("radius {}".format(radius))

    plt.show()

    return inverse_Pass




def process_factory(filename,filter_type):
    # try:
    #     Img_Process(filename,filter_type)
    # except:
    #     raise Exception("Unkown Error in process")

    # include complex number
    processed_img=Img_Process(filename,filter_type)


process_factory("er.bmp","guassHP")

