#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/16.
"""

class BaseMask(object):
    """
    根据 cv2 读进来的image 对象，colorLower,colorUpper数据获得感兴趣区域图片

    """

    def __init__(self, bkgImage, colorLower, colorUpper):
        self._image = bkgImage
        self._colorLower = colorLower
        self._colorUpper = colorUpper


    def getInterestImage(self):
        pass


