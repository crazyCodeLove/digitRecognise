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
    self._rollerBox: 黑底白字滚动轮区域在 self._image 中的四边对应元组
    self._lcdBox：LCD液晶屏区域在黑底白字表头下半部分的位置
    self._rollerImage: 黑底白字滚动轮区域图像，cv2 对象，BGR



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
        rollerBox = self.getRollerBox()
        rollerImage = ImageTool.getCropImageByBox(self._image,rollerBox)

        rollerImage = ImageTool.imageResize(rollerImage,self._desWidth,self._desHeight)

        if self._desImageDepth == 1:
            rollerImage = ImageTool.convertImgBGR2Gray(rollerImage)
        self._rollerImage = rollerImage


        return self._rollerImage

    def getRollerBox(self):
        """
        获得黑底白字滚轮区域box，并将该 box 赋值给 self._rollerBox
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

        rollerBox = (trollerBox[0],rollerBoxUp + 1,trollerBox[2],rollerBoxDown -1)
        getWidth = rollerBox[2]-rollerBox[0]
        if rollerBoxWidthMax < getWidth:
            rollerBox = (rollerBox[0],rollerBox[1],rollerBox[0] + rollerBoxWidthMax,rollerBox[3])

        right = int(rollerBox[2]*5/8) + 20
        rollerBox = (rollerBox[0], rollerBox[1], right, rollerBox[3])

        # ImageTool.showBoxInImageByBox(self._image,rollerBox)
        self._rollerBox = rollerBox

        return rollerBox

    def getLCDBox(self):
        """
        :return:
        """
        shape = self._image.shape
        mid = int(shape[0] / 2)
        splitImageBox = (0,mid,shape[1],shape[0])
        splitImageGray = ImageTool.getCropImageByBox(self._grayImage, splitImageBox)
        ImageTool.showImagePIL(splitImageGray)

        # splitImage = ImageTool.getCropImageByBox(self._image,splitImageBox)

        retval, otsuImage = ImageTool.getOTSUGrayImage(splitImageGray)
        otsuImage = ImageTool.convertImgGray2BGR(otsuImage)

        lower = (250, 250, 250)
        upper = (255, 255, 255)
        lcdBoxCorner = ImageTool.getInterestBoxCornerPointByColor(otsuImage, lower, upper)
        lcdBox = ImageTool.getBoxFromBoxCorner(lcdBoxCorner)
        self._lcdBox = lcdBox

        # ImageTool.showBoxInImageByBoxCornerPoint(splitImage,lcdBoxCorner,"lcd")
















    def getBarCodeArea(self):
        super(GasmeterStyle1,self).getBarCodeArea()

    def getLCDArea(self):
        lcdBox = self.getLCDBox()





    def setImage(self, image):
        """
        设置要处理的图片
        :param image: 是cv2读进来的图片对象
        提取表头黑色背景区域图片保存到 self._image 域中，方便以后提取条形码、液晶屏、黑底白字滚轮区域数字
        """
        if image is None:
            raise ValueError("image is None")

        image = ImageTool.preProcessImage(image)

        # ImageTool.showImagePIL(image)

        super(GasmeterStyle1, self).setImage(image)
        blackMask = MaskTool.getBlackMaskBGR()
        self._image = blackMask.getInterestImageAreaData(self._image)

        self._grayImage = ImageTool.convertImgBGR2Gray(self._image)


    def getDescription(self):
        desc = "背景不是黑色的，所有数字区域背景是黑色的。识别部分由三部分组成，1：条形码，2：黑底白字滚轮数字，3：液晶屏" \
               "黑底白字滚轮数字所在区域右边有红底白字。类型是style1"
        return desc

def testRollerBlock():
    import matplotlib.pyplot as plt
    import platform
    import cv2
    import os
    if "Windows" in platform.system():
        filename = r"D:\chengxu\python\project\digitRecognise\com\huitong\gasMeterv1\data\img\style1\2.jpg"
    elif "Linux" in platform.system():
        filename = r"/home/allen/work/digitRecognise/com/huitong/gasMeterv1/data/img/style1/000.jpg"

    style1 = GasmeterStyle1(desImageDepth=1)
    image = cv2.imread(filename)

    ImageTool.showImagePIL(image)

    style1.setImage(image)

    rollerBlackImage = style1.getRollerBlackArea()
    # rollerBlackImage = cv2.cvtColor(rollerBlackImage,cv2.COLOR_BGR2RGB)
    title = str(rollerBlackImage.shape) + os.path.basename(filename)
    # rollerBlackImage = ImageTool.convertImgBGR2Gray(rollerBlackImage)

    # ret,rollerBlackImage = ImageTool.getOTSUGrayImage(rollerBlackImage)

    plt.figure()
    plt.imshow(rollerBlackImage)
    plt.title(title)
    plt.show()

def testLCDBlock():
    import matplotlib.pyplot as plt
    import platform
    import cv2
    if "Windows" in platform.system():
        filename = r"D:\chengxu\python\project\digitRecognise\com\huitong\gasMeterv1\data\img\style1\11.jpg"
    elif "Linux" in platform.system():
        filename = r"/home/allen/work/digitRecognise/com/huitong/gasMeterv1/data/img/style1/000.jpg"

    style1 = GasmeterStyle1(desImageDepth=1)
    image = cv2.imread(filename)

    style1.setImage(image)
    style1.getLCDArea()



def test():
    # testRollerBlock()
    testLCDBlock()



if __name__ == "__main__":
    test()