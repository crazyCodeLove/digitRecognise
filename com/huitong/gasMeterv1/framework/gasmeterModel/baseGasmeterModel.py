#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/17.
"""

class BaseGasmeterModel(object):

    def setImage(self,image):
        """
        设置要处理的图片
        :param image: 是cv2读进来的图片对象
        """
        self._image = image

    @property
    def GetImage(self):
        return self._image

    def getDescription(self):
        """显示该模型的描述，是哪一种类型的燃气表"""
        pass

    def getRollerBlackArea(self):
        """获得滚轮黑底白字区域数据，并进行大小规整"""
        pass

    def getLCDArea(self):
        """获得 LCD 区域图片数据，并进行大小规整"""
        pass

    def getBarCodeArea(self):
        """获得条形码区域图片数据，并进行大小规整"""
        pass
