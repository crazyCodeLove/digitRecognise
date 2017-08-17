#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/17.
"""
from com.huitong.gasMeterv1.framework.gasmeterModel.baseGasmeterModel import BaseGasmeterModel
from com.huitong.gasMeterv1.framework.gasmeterModel.BaseMask import MaskTool
from com.huitong.gasMeterv1.framework.tool.ImageTool import ImageTool



class GasmeterStyle0(BaseGasmeterModel):
    """
    会对所有输出图片大小进行规整，默认是 64 * 128
    """

    def __init__(self, desWidth = None, desHeight = None):
        if desHeight is None:
            self._desHeight = 64
        else:
            self._desHeight = desHeight

        if desWidth is None:
            self._desWidth = 128
        else:
            self._desWidth = desWidth

    def setImage(self,image):
        """
        设置要处理的图片
        :param image: 是cv2读进来的图片对象
        """
        self.__image = image

    def getDescription(self):
        description = "背景不是黑色的，数字区域是黑底白字或红底白字，只需要提取出黑底白字滚轮区域的数字,红色区域在黑色区域右边。类型是 style0"
        return description

    def getRollerBlackArea(self):
        if self.__image is None:
            raise ValueError("应先通过setImage()函数设置image，然后获取感兴趣数据")

        blackMask = MaskTool.getBlackMask()
        blackImage = blackMask.getInterestImageAreaData(self.__image)

        redMask = MaskTool.getRedMask()
        redBox = redMask.getInterestBox(blackImage)

        if redBox is not None:
            rollerBlackArea = ImageTool.removeRightArea(blackImage,redBox[0])
        else:
            rollerBlackArea = blackImage

        rollerBlackArea = ImageTool.imageResize(rollerBlackArea, self._desWidth, self._desHeight)
        # rollerBlackArea = ImageTool.convertImgBGR2Gray(rollerBlackArea)

        return rollerBlackArea


def test():
    import cv2
    style = GasmeterStyle0()

    filename = r"D:\chengxu\python\project\digitRecognise\com\huitong\gasMeterv1\data\img\style0\8.jpg"
    filename = r'/home/allen/work/digitRecognise/com/huitong/gasMeterv1/data/img/style0/8.jpg'
    image = cv2.imread(filename)
    style.setImage(image)

    image = style.getRollerBlackArea()

    # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    ImageTool.showImagePIL(image)
    # ImageTool.showImageCv2(image)
    # image = cv2.equalizeHist(image)
    # ImageTool.showImageCv2(image)
    # ImageTool.showImagePIL(image)



if __name__ == "__main__":
    test()








