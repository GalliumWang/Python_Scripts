import cv2
import sys

try:
    filename=sys.argv[1]
except:
    print('help:\nwith the file name of jpg img as arg,and the img should be within the folder of this script :)')
    exit(1)

# must be jpg img
if not filename.endswith(".jpg"):
    raise Exception("must be jpg img")
elif len(filename.split("/"))>1 or len(filename.split("\\"))>1:
    raise Exception("img should be under same img with the script")

try:
    raw = cv2.imread(filename)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    
    # cv2.imshow('Original image',image)
    # cv2.imshow('Gray image', gray)
    
    cv2.imwrite(filename[:-4]+"_gray"+".jpg",gray)
except:
    raise Exception('Sorry,\nUnkown exception')

