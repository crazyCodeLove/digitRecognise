#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/16.
"""
import cv2
import platform
import numpy as np
import matplotlib.pyplot as plt

from com.huitong.gasMeterv1.framework.tool.filenameUtil import FileNameUtil
from PIL import Image


class ImageTool():
    """
    showGasmeterArea(filename):显示文件中燃气表数字所在区域。说明：燃气表数字应在图片居中位置，表头大小 <= 图片总大小的80%
    图片背景颜色最好是亮色系，不能是黑色。最好是白色。
    """

    @staticmethod
    def getBlackColorRangeBGR():
        """
        获得黑色 BGR 颜色范围，返回(lower, upper)元组，lower、upper 是 BGR 表示的元组
        """
        return ((0,0,0),(70,70,70))

    @staticmethod
    def getGrayColorRangeHSV():
        return ((0,0,46),(180,43,220))




    @staticmethod
    def getRedColorRangeBGR():
        """获得黑色 BGR 颜色范围，返回(lower, upper)元组，lower、upper 是 BGR 表示的元组"""
        return ((0,0,100),(60,45,255))



    @staticmethod
    def convertImgBGR2Gray(bgrimg):
        """
        将cv2 读进来的图像转换成灰度图像对象并返回
        :param bgrimg:
        """
        return cv2.cvtColor(bgrimg,cv2.COLOR_BGR2GRAY)

    @staticmethod
    def convertImgRGB2Gray(rgbimg):
        """

        :param rgbimg: 彩色图像，是RGB 图片
        """
        rgbimg = np.array(rgbimg)
        return cv2.cvtColor(rgbimg,cv2.COLOR_RGB2GRAY)

    @staticmethod
    def convertImgGray2BGR(grayImg):
        """
        将灰度图像转化为 cv2 的 BGR 图像对象，
        :param grayImg: 灰度图像
        """
        bgrImg = cv2.cvtColor(grayImg,cv2.COLOR_GRAY2BGR)
        return bgrImg



    @staticmethod
    def preProcessImage(img, width=None):
        """
        将输入图片缩放到指定大小，默认是800*800。裁剪所得图片中间80%，并恢复到指定大小
        :param img: 输入图片数据，cv2读取的 object 对象
        :param width: 指定图片的大小
        """
        if width is None:
            width = 800
        img = cv2.resize(img, (width, width), interpolation=cv2.INTER_CUBIC)

        boxhw = int(width * 0.8)
        starthw = int((width - boxhw) / 2)
        # cropImg = image[y1:y1+hight, x1:x1+width]图片裁剪
        img = img[starthw:starthw + boxhw, starthw:starthw + boxhw]

        img = cv2.resize(img, (width, width), interpolation=cv2.INTER_CUBIC)
        return img

    @staticmethod
    def showBoxInImageByBoxCornerPoint(image, boxCornerPoint, title=None):
        """
        在图片上根据矩形框的四个顶点坐标显示矩形框
        :param image: 通过cv2 读进来的图像对象
        :param boxCornerPoint: 矩形框的四个顶点
        """
        if title is None:
            title = "box"
        cimage = image.copy()
        cv2.drawContours(cimage,[boxCornerPoint],0,(0,255,0),2)
        cv2.imshow(title,cimage)

        k = cv2.waitKey(0)  # 无限期等待输入
        if k == 27:  # 如果输入ESC退出
            cv2.destroyAllWindows()

    @staticmethod
    def showBoxInImageByBox(image,box,title=None):
        """

        :param image: 通过cv2 读进来的图像对象
        :param box: 矩形框的 left, upper, right, and lower pixel coordinate tuple
        :param title:
        """
        boxCorner = ImageTool.getBoxCornerPointFromBox(box)
        ImageTool.showBoxInImageByBoxCornerPoint(image,boxCorner,title)

    @staticmethod
    def getInterestBoxCornerPointByColor(image, lower, upper):
        """
        获取感兴趣的颜色范围所在矩形框的四个顶点，如果图片中没有像素在范围内返回 None
        :param image: 通过cv2读进来的图片数据对象
        :param lower: # 颜色下限,数值按[b,g,r]排布
        :param upper: # 颜色上限,数值按[b,g,r]排布
        """
        # 根据阈值提取在区间范围内的数据，在范围内的数据为 255，其他数据置 0
        mask = cv2.inRange(image, lower, upper)
        if "Linux" in platform.system():
            cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            (erodeImg, cnts, hierarchy) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) == 0:
            return None

        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        # box里保存的是绿色矩形区域四个顶点的坐标。
        if "Linux" in platform.system():
            boxCornerPoint = np.int0(cv2.cv.BoxPoints(rect))
        else:
            boxCornerPoint = np.int0(cv2.boxPoints(rect))

        return boxCornerPoint

    @staticmethod
    def getBoxFromBoxCorner(boxCornerPoint):
        """
        根据 box 四个角的坐标获得 left, upper, right, and lower pixel coordinate tuple
        原点(0,0)在 left upper corner，x 轴向下，y 轴向右
        :param boxCorner:
        """
        Xs = [i[0] for i in boxCornerPoint]
        Ys = [i[1] for i in boxCornerPoint]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        return (x1, y1, x2, y2)

    @staticmethod
    def getBoxCornerPointFromBox(box):
        # cornerPoint = np.array(((box[0],box[1]),(box[2],box[1]),(box[2],box[4]),(box[0],box[3])))
        t = [[box[0],box[1]],
            [box[2],box[1]],
            [box[2],box[3]],
            [box[0],box[3]]]
        cornerPoint = np.array(t)
        return cornerPoint

    @staticmethod
    def getBoxWidth(box):
        return box[2] - box[0]

    @staticmethod
    def getBoxHeight(box):
        return box[3] - box[1]

    @staticmethod
    def getCropImageByBoxCornerPoint(image, boxCornerPoint):
        """
        获取裁剪图片，根据要裁剪的矩形四个顶点
        :param image 是cv2读进来的图片对象
        :param boxCornerPoint: 要裁剪的矩形四个顶角坐标
        """
        cimage = image.copy()
        box = ImageTool.getBoxFromBoxCorner(boxCornerPoint)
        height = ImageTool.getBoxHeight(box)
        width = ImageTool.getBoxWidth(box)
        left = box[0]
        upper = box[1]

        cropImg = cimage[upper:upper + height, left:left + width]
        return cropImg

    @staticmethod
    def getCropImageByBox(image,box):
        """
        获取裁剪图片，根据要裁剪的矩形 Box
        :param image: 是cv2读进来的图片对象
        :param box: 要裁剪的矩形,使用 left, upper, right, and lower pixel coordinate tuple 来表示
        """
        left = box[0]
        upper = box[1]
        width = box[2] - box[0]
        height = box[3] - box[1]
        cropImg = image[upper:upper + height, left:left + width]
        return cropImg

    @staticmethod
    def repeatImage2Tensor(image, batchsize):
        """
        将图片对象转换成一维 tensor，然后重复 batchsize 次，返回。主要用于预测数据
        :param image: 是cv2读进来的图片对象
        :param batchsize:
        """
        image = np.array(image, dtype=np.float32).flatten()
        image = image.reshape((-1, image.shape[0]))
        batch_data = np.repeat(image, batchsize, 0)
        return batch_data

    @staticmethod
    def removeRightArea(image,startIndex):
        """
        以 startIndex 为界分成左右两半部分，移除图像右边部分
        :param image: 是cv2读进来的图片对象
        :param startIndex: 要移除的图像水平轴开始边界
        rows, cols, depth = image.shape
        """
        shape = image.shape
        box = (0,0,startIndex,shape[0])
        cropImg = ImageTool.getCropImageByBox(image,box)
        return cropImg

    @staticmethod
    def removeLeftArea(image,startIndex):
        """
        以 startIndex 为界分成左右两半部分，移除图像右边部分
        :param image: 是cv2读进来的图片对象
        :param startIndex:
        rows, cols, depth = image.shape
        """
        shape = image.shape
        box = (startIndex,0,shape[1],shape[0])
        cropImg = ImageTool.getCropImageByBox(image, box)
        return cropImg

    @staticmethod
    def removeUpArea(image,startIndex):
        """
        以 startIndex 为界分成上下两半部分，移除图像上边部分
        :param image: 是cv2读进来的图片对象
        :param startIndex:
        rows, cols, depth = image.shape
        """
        shape = image.shape
        box = (0, startIndex, shape[1], shape[0])
        cropImg = ImageTool.getCropImageByBox(image, box)
        return cropImg

    @staticmethod
    def removeDownArea(image,startIndex):
        """
        以 startIndex 为界分成上下两半部分，移除图像上边部分
        :param image: 是cv2读进来的图片对象
        :param startIndex:
        rows, cols, depth = image.shape
        """
        shape = image.shape
        box = (0, 0, shape[1], startIndex)
        cropImg = ImageTool.getCropImageByBox(image, box)
        return cropImg

    @staticmethod
    def imageResize(image, width, height):
        """
        :param image: 是cv2读进来的图片对象
        :param width: 目标图像宽度
        :param height: 目标图像高度
        cv2.resize(image, (BoxWidth, BoxHeight))
        """
        image = cv2.cvtColor(np.array(image),cv2.COLOR_BGR2RGB)

        image = Image.fromarray(np.array(image))
        image = ImageTool.imageResizePIL(image,width,height)
        image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
        return image

    @staticmethod
    def imageResizeCV2(image,width,height):
        """
        :param image: 是cv2读进来的图片对象
        :param width: 目标图像宽度
        :param height: 目标图像高度
        cv2.resize(image, (BoxWidth, BoxHeight))
        """
        image = cv2.resize(image,(width,height),interpolation=cv2.INTER_CUBIC)
        return image

    @staticmethod
    def imageResizePIL(image,width,height):
        """

        :param image: PIL.Image 读进来的图片对象
        :param width: 目标图像宽度
        :param height: 目标图像高度
        :return:
        """
        image = image.resize((width, height), Image.BICUBIC)
        return image


    @staticmethod
    def getOTSUGrayImage(image):
        """
        对灰度图像进行 大津算法
        :param image: 灰度图像
        返回值：-> retval, dst
        """
        return cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)  # 只支持灰度图像的STSU




    @staticmethod
    def showImagePIL(image,title = None):
        """
        通过 PIL 显示图片
        :param image: cv2读进来的图片对象
        :param title: 标题
        """
        if title is None:
            title = str(image.shape)
        plt.figure()
        plt.imshow(image)
        plt.title(title)
        plt.show()

    @staticmethod
    def showImageCv2(image,title = None):
        """
        通过 cv2 显示图片
        :param image: cv2读进来的图片对象
        :param title:  标题
        """
        if title is None:
            title = str(image.shape)
        cv2.imshow(title,image)
        k = cv2.waitKey(0)

        if k == 27:
            cv2.destroyAllWindows()

    @staticmethod
    def showHistBGR(image,title = None):
        """
        显示 cv2 读进来的彩色图像直方图
        :param image: cv2读进来的图片对象
        """
        if title is None:
            title = image.shape
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.title(title)
        plt.show()

    @staticmethod
    def showHistGray(image,title=None):
        """
        显示 cv2 读进来的灰度图像直方图
        :param image: cv2读进来的图片对象
        :param title:
        """
        if title is None:
            title = image.shape
        plt.hist(image.ravel(), 255, [0, 256])
        plt.title(title)
        plt.show()











def testShowInterestAreaBox():
    import platform
    imgdirname = ["data", "img", "style1"]
    if "Windows" in platform.system():
        imgdirname = FileNameUtil.getDirname(r"D:\chengxu\python\project\digitRecognise\com\huitong\gasMeterv1", imgdirname)
    elif "Linux" in platform.system():
        imgdirname = FileNameUtil.getDirname(r"/home/allen/work/digitRecognise/com/huitong/gasMeterv1", imgdirname)

    pattern = r'.*\.jpg$'
    filelist = FileNameUtil.getPathFilenameList(imgdirname, pattern)
    interestColorLower = (0,0,0)
    interestColorUpper = (70,70,70)

    for filename in filelist:
        image = cv2.imread(filename)
        image = ImageTool.preProcessImage(image)
        boxCornerPoint = ImageTool.getInterestBoxCornerPointByColor(image,interestColorLower,interestColorUpper)

        if boxCornerPoint is not None:
            box = ImageTool.getBoxFromBoxCorner(boxCornerPoint)
            boxCornerPoint = ImageTool.getBoxCornerPointFromBox(box)
            title = "box area, %s" % FileNameUtil.getFilenameFromFullFilepathname(filename)
            ImageTool.showBoxInImageByBoxCornerPoint(image, boxCornerPoint, title)
        else:
            print("no color in range" + str(interestColorLower) + str(interestColorUpper))



def testShowInterestAreaDataStyle0():
    import platform
    imgdirname = ["data", "img", "style0"]

    if "Windows" in platform.system():
        imgdirname = FileNameUtil.getDirname(r"D:\chengxu\python\project\digitRecognise\com\huitong\gasMeterv1", imgdirname)
    elif "Linux" in platform.system():
        imgdirname =  FileNameUtil.getDirname(r"/home/allen/work/digitRecognise/com/huitong/gasMeterv1",imgdirname)

    pattern = r'.*\.jpg$'
    filelist = FileNameUtil.getPathFilenameList(imgdirname, pattern)
    bkgLower,bkgUpper = ImageTool.getBlackColorRangeBGR()
    redLower,redUpper = ImageTool.getRedColorRangeBGR()

    for filename in filelist:

        print(filename)

        image = cv2.imread(filename)
        image = ImageTool.preProcessImage(image)
        blackBackgroundBoxCornerPoint = ImageTool.getInterestBoxCornerPointByColor(image, bkgLower, bkgUpper)

        if blackBackgroundBoxCornerPoint is None:
            raise ValueError("no color is black")

        bkgBox = ImageTool.getBoxFromBoxCorner(blackBackgroundBoxCornerPoint)

        bkgimage = ImageTool.getCropImageByBoxCornerPoint(image, blackBackgroundBoxCornerPoint)

        redBoxCornerPoint = ImageTool.getInterestBoxCornerPointByColor(bkgimage, redLower, redUpper)

        if redBoxCornerPoint is not None:
            redBox = ImageTool.getBoxFromBoxCorner(redBoxCornerPoint)

            box = (0,0,redBox[0],bkgBox[3]-bkgBox[1])
            interestImage = ImageTool.getCropImageByBox(bkgimage,box)
            interestImage = ImageTool.imageResize(interestImage,128,64)
        else:
            interestImage = bkgimage


        # title = "box area, %s, background" % FileNameUtil.getFilenameFromFullFilepathname(filename)
        # bkgimage = cv2.cvtColor(bkgimage,cv2.COLOR_BGR2RGB)
        #
        # plt.figure()
        # plt.title(title)
        # plt.imshow(bkgimage)
        # plt.show()

        title = "box area, %s, digit area,%s" % (FileNameUtil.getFilenameFromFullFilepathname(filename),str(interestImage.shape))
        grayImage = cv2.cvtColor(interestImage,cv2.COLOR_BGR2GRAY)
        plt.figure()
        plt.title(title)
        plt.imshow(grayImage)
        plt.show()

        # cv2.imshow("gray", grayImage)
        # k = cv2.waitKey(0)
        # if k == 27:
        #     cv2.destroyAllWindows()

        # interestImage = cv2.cvtColor(interestImage, cv2.COLOR_BGR2RGB)
        #
        #
        # plt.figure()
        # plt.title(title)
        # plt.imshow(interestImage)
        # plt.show()

def testShowInterestAreaDataStyle1():
    import platform
    imgdirname = ["data", "img", "style1"]

    if "Windows" in platform.system():
        imgdirname = FileNameUtil.getDirname(r"D:\chengxu\python\project\digitRecognise\com\huitong\gasMeterv1",
                                             imgdirname)
    elif "Linux" in platform.system():
        imgdirname = FileNameUtil.getDirname(r"/home/allen/work/digitRecognise/com/huitong/gasMeterv1", imgdirname)

    pattern = r'.*\.jpg$'
    filelist = FileNameUtil.getPathFilenameList(imgdirname, pattern)

    redLower, redUpper = ImageTool.getRedColorRangeBGR()

    blackLower,blackUpper= ImageTool.getBlackColorRangeBGR()

    for filename in filelist:

        print(filename)

        image = cv2.imread(filename)
        image = ImageTool.preProcessImage(image)

        cornerPoint = ImageTool.getInterestBoxCornerPointByColor(image,blackLower,blackUpper)
        if cornerPoint is None:
            raise ValueError("none black data")

        interestImage = ImageTool.getCropImageByBoxCornerPoint(image,cornerPoint)

        shape = interestImage.shape
        mid =int(shape[0]/2)
        quarter = int(shape[0]/4)
        upbox = (0,0,shape[1],mid)
        upImage = ImageTool.getCropImageByBox(interestImage,upbox)

        title = "box area, %s, digit area,%s" % (
        FileNameUtil.getFilenameFromFullFilepathname(filename), str(upImage.shape))
        upImageGray = cv2.cvtColor(upImage, cv2.COLOR_BGR2GRAY)

        # equalizeImage = cv2.equalizeHist(upImageGray)  # 灰度图像直方图均衡化
        # retval,otsuImage = cv2.threshold(equalizeImage,0,255,cv2.THRESH_OTSU) #只支持灰度图像的STSU

        shape = upImageGray.shape
        box = (70,quarter,170,shape[0])
        splitLineImage = ImageTool.getCropImageByBox(upImageGray,box)
        # equalizeImage = cv2.equalizeHist(splitLineImage)
        retval, otsuImage = cv2.threshold(splitLineImage, 0, 255, cv2.THRESH_OTSU)






        ImageTool.showImagePIL(upImageGray,title)
        # ImageTool.showHistGray(upImageGray,title)
        # ImageTool.showImagePIL(equalizeImage,title)
        # ImageTool.showHistGray(equalizeImage,title)
        # ImageTool.showImagePIL(otsuImage,title)


        # ImageTool.showImagePIL(splitLineImage,title)
        # ImageTool.showImagePIL(otsuImage, title)
        #
        otsuImage = cv2.cvtColor(otsuImage,cv2.COLOR_GRAY2BGR)
        lower = (250,250,250)
        upper = (255,255,255)
        tcorner = ImageTool.getInterestBoxCornerPointByColor(otsuImage,lower,upper)
        ImageTool.showBoxInImageByBoxCornerPoint(otsuImage, tcorner, title)














if __name__ == "__main__":
    testShowInterestAreaBox()
    # testShowInterestAreaDataStyle0()
    # testShowInterestAreaDataStyle1()

