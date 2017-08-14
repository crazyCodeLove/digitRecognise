#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/14.
"""
from PIL import Image
from com.huitong.gasMeterv1.filenameUtil import FileNameUtil
import numpy as np
import matplotlib.pyplot as plt

def fun1():
    a = None
    b = 200
    if a:
        print("a is true")
    else:
        print("a is false")

    if b:
        print("b is true")
    else:
        print("b is false")

def fun2():
    imgdirname = ["data","img"]
    imgdirname = FileNameUtil.getDirname(FileNameUtil.getBasedirname(__file__),imgdirname)
    imgdirname = FileNameUtil.getPathJoin(imgdirname,"pic1.jpg")

    img = Image.open(imgdirname)

    plt.figure()
    plt.imshow(img)
    plt.title("ori")
    plt.show()

    pastImg = np.zeros([20,40],dtype=np.uint8)
    plt.figure()
    plt.imshow(pastImg)
    plt.title("past")
    plt.show()

    pastImg = Image.fromarray(pastImg)
    # left, upper, right, and lower pixel coordinate
    img.paste(pastImg,(40,0,80,20))
    plt.figure()
    plt.imshow(img)
    plt.title("after")
    plt.show()


def test():
    fun2()

if __name__ == "__main__":
    test()
