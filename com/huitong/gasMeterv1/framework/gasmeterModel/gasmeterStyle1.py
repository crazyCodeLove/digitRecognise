#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/18.
"""
from com.huitong.gasMeterv1.framework.gasmeterModel.baseGasmeterModel import BaseGasmeterModel

class GasmeterStyle1(BaseGasmeterModel):

    def getRollerBlackArea(self):
        super().getRollerBlackArea()

    def getBarCodeArea(self):
        super().getBarCodeArea()

    def getLCDArea(self):
        super().getLCDArea()

    def setImage(self, image):
        super().setImage(image)

    def getDescription(self):
        desc = "背景不是黑色的，所有数字区域背景是黑色的。识别部分由三部分组成，1：条形码，2：黑底白字滚轮数字，3：液晶屏" \
               "黑底白字滚轮数字所在区域右边有红底白字。类型是style1"
        return desc