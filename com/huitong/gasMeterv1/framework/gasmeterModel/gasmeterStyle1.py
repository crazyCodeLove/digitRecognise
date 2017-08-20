#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/18.
"""
from com.huitong.gasMeterv1.framework.gasmeterModel.baseGasmeterModel import BaseGasmeterModel
from com.huitong.gasMeterv1.framework.gasmeterModel.BaseMask import MaskTool
from com.huitong.gasMeterv1.framework.tool.ImageTool import ImageTool

class GasmeterStyle1(BaseGasmeterModel):
    """
    self._image: cv2 读进来的黑底白字的表头图像对象，是彩色图像
    self._grayImage: cv2 读进来的黑底白字的表头图像对象，对应的黑白图像
    self._rollerBox:

    """

    def __init__(self, desWidth = None, desHeight = None, desImageDepth = 3):
        if desHeight is None:
            self._desHeight = 64
        else:
            self._desHeight = desHeight

        if desWidth is None:
            self._desWidth = 128
        else:
            self._desWidth = desWidth

        self._desImageDepth = desImageDepth



    def getRollerBlackArea(self):
        self.getRollerBox()
        rollerImage = ImageTool.getCropImageByBox(self._image,self._rollerBox)


        if self._desImageDepth == 1:
            return ImageTool.convertImgBGR2Gray(self._image)


        return self._image

    def getRollerBox(self):
        """
        获得黑底白字滚轮区域box，并将该 box 赋值给 self._rollerBox
        :return:
        """
        rollerBoxWidthMax = 425

        shape = self._image.shape
        mid = int(shape[0] / 2)
        quarter = int(shape[0] / 4)
        splitImageBox = (70,quarter,170,mid)
        splitImageGray = ImageTool.getCropImageByBox(self._grayImage,splitImageBox)

        retval, otsuImage = ImageTool.getOTSUGrayImage(splitImageGray)
        otsuImage = ImageTool.convertImgGray2BGR(otsuImage)

        lower = (250,250,250)
        upper = (255,255,255)
        splitBoxCorner = ImageTool.getInterestBoxCornerPointByColor(otsuImage,lower,upper)
        splitBox = ImageTool.getBoxFromBoxCorner(splitBoxCorner)
        rollerBoxUp = int(quarter + splitBox[1])
        rollerBoxDown = int(quarter + splitBox[3])

        trollerBox = (0,rollerBoxUp,shape[1],rollerBoxDown)
        rollerImageGray = ImageTool.getCropImageByBox(self._grayImage,trollerBox)
        retval, otsuImage = ImageTool.getOTSUGrayImage(rollerImageGray)
        otsuImage = ImageTool.convertImgGray2BGR(otsuImage)
        trollerBoxCorner = ImageTool.getInterestBoxCornerPointByColor(otsuImage, lower, upper)
        trollerBox = ImageTool.getBoxFromBoxCorner(trollerBoxCorner)

        rollerBox = (trollerBox[0],rollerBoxUp - 2,trollerBox[2],rollerBoxDown + 8)
        getWidth = rollerBox[2]-rollerBox[0]
        if rollerBoxWidthMax < getWidth:
            rollerBox = (rollerBox[0],rollerBox[1],rollerBox[0] + rollerBoxWidthMax,rollerBox[3])

        right = int(rollerBox[2]*5/8) + 20
        rollerBox = (rollerBox[0], rollerBox[1], right, rollerBox[3])

        ImageTool.showBoxInImageByBox(self._image,rollerBox)
        self._rollerBox = rollerBox

        return rollerBox








    def getBarCodeArea(self):
        super().getBarCodeArea()

    def getLCDArea(self):
        super().getLCDArea()

    def setImage(self, image):
        """
        设置要处理的图片
        :param image: 是cv2读进来的图片对象
        提取表头黑色背景区域图片保存到 self._image 域中，方便以后提取条形码、液晶屏、黑底白字滚轮区域数字
        """
        image = ImageTool.preProcessImage(image)
        super(GasmeterStyle1, self).setImage(image)
        blackMask = MaskTool.getBlackMaskBGR()
        self._image = blackMask.getInterestImageAreaData(self._image)
        self._grayImage = ImageTool.convertImgBGR2Gray(self._image)


    def getDescription(self):
        desc = "背景不是黑色的，所有数字区域背景是黑色的。识别部分由三部分组成，1：条形码，2：黑底白字滚轮数字，3：液晶屏" \
               "黑底白字滚轮数字所在区域右边有红底白字。类型是style1"
        return desc



def test():
    import matplotlib.pyplot as plt
    import platform
    import cv2
    import os
    if "Windows" in platform.system():
        filename = r"D:\chengxu\python\project\digitRecognise\com\huitong\gasMeterv1\data\img\style1\3.jpg"
    elif "Linux" in platform.system():
        filename = ""

    style1 = GasmeterStyle1(desImageDepth=3)
    image = cv2.imread(filename)
    style1.setImage(image)



    image = style1.getRollerBlackArea()
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    title = str(image.shape) + os.path.basename(filename)

    # plt.figure()
    # plt.imshow(image)
    # plt.title(title)
    # plt.show()


if __name__ == "__main__":
    test()