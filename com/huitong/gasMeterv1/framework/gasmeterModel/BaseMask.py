#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/16.
"""
from com.huitong.gasMeterv1.framework.tool.ImageTool import ImageTool

class BaseMask(object):
    """
    根据 cv2 读进来的image 对象，colorLower,colorUpper数据获得感兴趣区域图片
    用来获得感兴趣区域图片数据、感兴趣区域边框、感兴趣区域四角

    """

    def __init__(self, colorLower, colorUpper):
        """

        :param colorLower: BGR 颜色范围，是 BGR 表示的元组，表示颜色范围下限
        :param colorUpper: BGR 颜色范围，是 BGR 表示的元组，表示颜色范围上限
        """
        self._colorLower = colorLower
        self._colorUpper = colorUpper

    def getInterestBoxCornerPoint(self,bkgImage):
        """
        获得感兴趣的图片区域所在矩形的四个顶角坐标，如果图片中没有在所选区域范围内，返回 None
        :param bkgImage: 通过cv2 读进来的图像对象
        """
        boxCornerPoint = ImageTool.getInterestBoxCornerPointByColor(bkgImage, self._colorLower, self._colorUpper)
        return boxCornerPoint

    def getInterestBox(self,bkgImage):
        """
        获得感兴趣的区域box，box 是由 (left, upper, right, lower ) pixel coordinate tuple 来表示
        如果没有返回 None
        :param bkgImage: 通过cv2 读进来的图像对象
        """
        boxCornerPoint = self.getInterestBoxCornerPoint(bkgImage)
        if boxCornerPoint is None:
            return None
        box = ImageTool.getBoxFromBoxCorner(boxCornerPoint)
        return box

    def getInterestImageAreaData(self, bkgImage):
        """
        获得图片中感兴趣的颜色范围区域数据，如果没有返回 None
        :param bkgImage: 通过cv2 读进来的图像对象
        """
        boxCornerPoint = self.getInterestBoxCornerPoint(bkgImage)
        if boxCornerPoint is None:
            return None
        image = ImageTool.getCropImageByBoxCornerPoint(bkgImage, boxCornerPoint)

        return image





class MaskTool(object):

    @staticmethod
    def getBlackMaskBGR():
        """获得黑色区域的 Mask"""
        blkLower,blkUpper = ImageTool.getBlackColorRangeBGR()
        blackMask = BaseMask(blkLower,blkUpper)
        return blackMask

    @staticmethod
    def getRedMaskBGR():
        """获得红色区域的 Mask"""
        redLower, redUpper = ImageTool.getRedColorRangeBGR()
        redMask = BaseMask(redLower,redUpper)
        return redMask






