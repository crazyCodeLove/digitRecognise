#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/14.
"""
import platform

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

from com.huitong.gasMeterv1.framework.tool.filenameUtil import FileNameUtil
from com.huitong.gasMeterv1.framework.tool.ImageTool import ImageTool
from captcha.image import ImageCaptcha

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

def fun3():
    """
    提取某一区间的图片
    :return:
    """
    image = cv2.imread(r'D:\chengxu\python\project\digitRecognise\com\huitong\gasMeterv1\data\img\1.jpg')
    # color = [((0, 0, 0),(70, 70, 70))]  # 黄色范围~这个是我自己试验的范围，可根据实际情况自行调整~注意：数值按[b,g,r]排布
    # for (lower, upper) in color:
    #     lower = np.array(lower, dtype=np.uint8)  # 颜色下限
    #     upper = np.array(upper, dtype=np.uint8)  # 颜色上限
    # lower = (0,0,0) # 颜色下限,数值按[b,g,r]排布
    # upper = (70,70,70) # 颜色上限

    lower = (0, 70, 70)  # 颜色下限,数值按[b,g,r]排布
    upper = (100, 255, 255)  # 颜色上限

    # 根据阈值提取在区间范围内的数据，在范围内的数据为 255，其他数据置 0
    mask = cv2.inRange(image, lower, upper)

    if "Linux" in platform.system():
        cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        (erodeImg, cnts, hierarchy) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    # box里保存的是绿色矩形区域四个顶点的坐标。
    if "Linux" in platform.system():
        boxCornerPoint = np.int0(cv2.cv.BoxPoints(rect))
    else:
        boxCornerPoint = np.int0(cv2.boxPoints(rect))

    print(boxCornerPoint)

    plt.figure()
    plt.imshow(mask)
    plt.show()

    return boxCornerPoint



def fun4():
    filename = r"D:\chengxu\python\project\digitRecognise\com\huitong\gasMeterv1\data\img\gasmeterHead\9.jpg"
    image = Image.open(filename)
    plt.figure()
    plt.imshow(image)
    plt.show()

def fun5():
    filename = r"D:\chengxu\python\project\digitRecognise\com\huitong\gasMeterv1\data\img\style1\2.jpg"
    image = cv2.imread(filename)
    ImageTool.showImageCv2(image)

def fun6():
    path = os.path.dirname(__file__)
    print(path)
    path = os.path.dirname(path)
    print(path)
    path = os.path.dirname(path)
    print(path)






def test():
    fun5()

if __name__ == "__main__":
    test()
