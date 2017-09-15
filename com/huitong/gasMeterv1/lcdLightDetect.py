#!/usr/bin/env python
#encoding=utf-8

"""
检测 lcd 液晶屏点阵是否全部点亮
@author ZHAOPENGCHENG on 2017/9/4.
"""
import cv2

from com.huitong.gasMeterv1.framework.gasmeterModel.BaseMask import MaskTool
from com.huitong.gasMeterv1.framework.tool.ImageTool import ImageTool

import platform

class LCDLightDetect(object):

    @staticmethod
    def getMinBox(lcdBoxCorner):
        Xs = [i[0] for i in lcdBoxCorner]
        Ys = [i[1] for i in lcdBoxCorner]
        Xs.sort()
        Ys.sort()
        if Xs[2] - Xs[1] > 4:
            Xs[1] = Xs[1] + 2
            Xs[2] = Xs[2] - 2
        if Ys[2] - Ys[1] >4:
            Ys[1] = Ys[1] + 2
            Ys[2] = Ys[2] - 2
        box = (Xs[1], Ys[1], Xs[2], Ys[2])
        return box

    @staticmethod
    def getLCDAreaData(image):
        """
        获取 LCD 区域图片数据
        :param image: cv2 读进来的图片对象
        """
        image = ImageTool.preProcessImage(image)
        blackMask = MaskTool.getBlackMaskBGR()
        image = blackMask.getInterestImageAreaData(image)

        shape = image.shape
        grayImage = ImageTool.convertImgBGR2Gray(image)
        mid = int(shape[0] / 2)
        splitImageBox = (0, mid, shape[1], shape[0])
        splitImageGray = ImageTool.getCropImageByBox(grayImage, splitImageBox)
        splitImage = ImageTool.getCropImageByBox(image, splitImageBox)

        # 显示剪切的 lcd 屏所在的下半屏灰度图
        ImageTool.showImagePIL(splitImageGray, "splitImageGray")

        retval, otsuImage = ImageTool.getOTSUGrayImage(splitImageGray)
        otsuImage = ImageTool.convertImgGray2BGR(otsuImage)

        lower = (250, 250, 250)
        upper = (255, 255, 255)
        lcdBoxCorner = ImageTool.getInterestBoxCornerPointByColor(otsuImage, lower, upper)
        lcdBox = LCDLightDetect.getMinBox(lcdBoxCorner)
        # ImageTool.showBoxInImageByBox(splitImage, lcdBox)

        lcdImage = ImageTool.getCropImageByBox(splitImage,lcdBox)
        return lcdImage

    @staticmethod
    def lcdLighted(lcdImage):
        """
        检测lcd 区域图片是否点亮，如果亮返回True，如果不亮返回False
        :param lcdImage: cv2 读进来的图片对象
        """
        lower = (0, 0, 0)
        upper = (50, 50, 50)

        blackBoxCorner = ImageTool.getInterestBoxCornerPointByColor(lcdImage,lower,upper)
        # 如果获得的 lcd 区域有黑色区域，说明没有点亮。
        if blackBoxCorner is not None:
            return False


        lcdImageGray = ImageTool.convertImgBGR2Gray(lcdImage)


        retval, otsuImage = ImageTool.getOTSUGrayImage(lcdImageGray)
        otsuImage = ImageTool.convertImgGray2BGR(otsuImage)


        notLcdBoxCorner = ImageTool.getInterestBoxCornerPointByColor(otsuImage, lower, upper)
        ImageTool.showImagePIL(lcdImage, "lcdimage")

        return notLcdBoxCorner is None




    def detectLight(self,image):
        """
        :param image: cv2 读进来的图片对象
        如果 LCD 区域有不亮的返回 False, 否则返回 True
        """
        lcdImage = LCDLightDetect.getLCDAreaData(image)
        return LCDLightDetect.lcdLighted(lcdImage)




def testLCDLightDetect(filename):
    lcdDetect = LCDLightDetect()
    image = cv2.imread(filename)
    print(lcdDetect.detectLight(image))




def test():
    if "Windows" in platform.system():
        filename = r"D:\chengxu\python\project\digitRecognise\com\huitong\gasMeterv1\data\img\style1\2.jpg"
    elif "Linux" in platform.system():
        filename = r"/home/allen/work/digitRecognise/com/huitong/gasMeterv1/data/img/style1/1.jpg"
    testLCDLightDetect(filename)


if __name__ == "__main__":
    test()




