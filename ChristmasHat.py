# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 14:56:57 2017

@author: liurole

Merry Christmas

"""

import cv2
import sys
import os.path
import getopt
from PIL import Image

def detect(filename, cascade_file):
    if not os.path.isfile(filename):
        raise RuntimeError("%s: not found" % filename)
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)
    
    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.011,
                                     minNeighbors = 5,
                                     minSize = (12, 12))
    return faces, image


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], 'hs:d:', [ 'help', 'src =', 'dest =' ])
    
    src = './demo.jpg'
    dest = './微信官方.jpg'
    hat = './hat.png'
    cascade_file = './lbpcascade.xml'
    
    for key, value in opts:
        if key in ['-h', '--help']:
            print('ChristmasHat')
            print('参数定义：')
            print('-h, --help\t显示帮助')
            print('-s, --src\t来源图像')
            print('-d, --dest\t目标图像')
            sys.exit(0)
        if key in ['-s', '--src']:
            src = value
        if key in ['-d', '--dest']:
            dest = value

    print('ChristmasHat\tFrom: ', src, '\tTo: ', dest) 
    
    faces, image = detect(src, cascade_file)
    hat_img = Image.open(hat)
    hat_img = hat_img.convert('RGBA')
    human_img = Image.open(src)
    human_img = human_img.convert("RGBA")
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("AnimeFaceDetect", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    for (x, y, w, h) in faces:
        y -= 10
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(y, x, y + h, x + w))
        
        hat_img = hat_img.resize( (h, w) )#convert size of hat
        hat_region = hat_img
        #hat_region = hat_region.rotate(6)
    
        human_region = ( x, y - h//2, x + w, y - h//2 + h )
    
        human_img.paste(hat_region, human_region,mask=hat_img)
        
    human_img.show()
    