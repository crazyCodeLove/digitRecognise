#!/usr/bin/env python
#encoding=utf-8

"""
检测 lcd 液晶屏点阵是否全部点亮
@author ZHAOPENGCHENG on 2017/9/4.
"""
from com.huitong.gasMeterv1.framework.tool.ImageTool import ImageTool

class LCDLightDetect(object):

    def detectLight(self,image):
        """

        :param image: cv2 读进来的图片对象
        """
        shape = image.shape
        grayImage = ImageTool.convertImgBGR2Gray(image)
        mid = int(shape[0] / 2)
        splitImageBox = (0, mid, shape[1], shape[0])
        splitImageGray = ImageTool.getCropImageByBox(grayImage, splitImageBox)
        ImageTool.showImagePIL(splitImageGray)

        retval, otsuImage = ImageTool.getOTSUGrayImage(splitImageGray)
        otsuImage = ImageTool.convertImgGray2BGR(otsuImage)

        lower = (250, 250, 250)
        upper = (255, 255, 255)
        lcdBoxCorner = ImageTool.getInterestBoxCornerPointByColor(otsuImage, lower, upper)
        lcdBox = ImageTool.getBoxFromBoxCorner(lcdBoxCorner)
        self._lcdBox = lcdBox





