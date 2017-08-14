#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/11.
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random, time
from com.huitong.gasMeterv1.CaptchaTool import ImageCaptcha
import cv2
from com.huitong.gasMeterv1.filenameUtil import FileNameUtil
import platform


class GenDigitPicture():
    """
    用于产生数字验证码、数字表头图片、模拟摄像头采集的图片
    :param captchaCharacterLength:验证码上字符个数
    :param captchaBoxWidth：验证码图片宽度
    :param captchaBoxHeight：验证码图片高度
    :param batchSize：生成一批次数量，默认值是 64
    :param charset：生成验证码的字符集，默认是['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    """


    def __init__(self, captchaCharacterLength, captchaBoxWidth, captchaBoxHeight,
                 charset = None, backgroundColor = None, fontColor = None):
        self._picCharacterLength = captchaCharacterLength
        self._picBoxWidth = captchaBoxWidth
        self._picBoxHeight = captchaBoxHeight

        if charset is None:
            self._charset = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        else:
            self._charset = charset

        self._backgroundColor = backgroundColor
        self._fontColor = fontColor

    @property
    def ImageHeight(self):
        return self._picBoxHeight

    @property
    def ImageWidth(self):
        return self._picBoxWidth

    @property
    def ImageDepth(self):
        return 3

    @property
    def CharSet(self):
        return self._charset

    @property
    def Max_Character_Length(self):
        return self._picCharacterLength


    def _random_text(self):
        """生成指定长度的随机字符串
        """
        text = []
        for i in range(self._picCharacterLength):
            c = random.choice(self._charset)
            text.append(c)
        return text

    def get_compose_gasmeter_text_and_image(self, gasmeterImg):
        """
        根据原始燃气表图片对象，生成一张合成图片和对应的数字字符串
        返回数字字符串和图片数据
        :param gasmeterImg: cv2读进来的燃气表图片对象
        """
        image = ImageCaptcha(width=self._picBoxWidth, height=self._picBoxHeight,
                             backgroundColor=self._backgroundColor, fontColor=self._fontColor)
        #随机选出字符串
        captcha_text = self._random_text()
        captcha_text = ''.join(captcha_text)

        #由字符串生成验证码
        captcha = image.generate(captcha_text)
        captcha_image = Image.open(captcha)

        # 获取表头数字区域
        # gasmeterImg = cv2.imread(gasmeterFilename)
        imgTemp = gasmeterImg.copy()
        imgTemp = ImageTool.convertCv2ColorImg2Gray(imgTemp)
        boxCornerPoint = ImageTool.getGasmeterRectBoxCornerPoint(grayImg=imgTemp)
        box = ImageTool.getBoxFromBoxCorner(boxCornerPoint)
        digitAreaWidth = ImageTool.getBoxWidth(box)
        digitAreaWidth = int(digitAreaWidth*5/8) + 20
        digitAreaHeight = ImageTool.getBoxHeight(box)
        # 如果截取的 #（5,8）表头宽度比验证码宽度长，则验证码 resize 到 #(5,8)表头大小
        # #（5,8）表示燃气表数字有 8 位，其中5 位是要识别的黑底白字
        if digitAreaWidth > self._picBoxWidth:
            captcha_image = captcha_image.resize((digitAreaWidth,digitAreaHeight))
            box = (box[0],box[1],box[0] + digitAreaWidth, box[1] + digitAreaHeight)
        else:
            box = (box[0],box[1],box[0] + captcha_image.width, box[1] + captcha_image.height)


        # 读取原图片，预处理，以便黏贴验证码
        # oriImg = gasmeterImg.copy()

        oriImg = gasmeterImg.copy()
        oriImg = ImageTool.preProcessImage(oriImg)
        oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
        data = np.array(oriImg)
        oriImg = Image.fromarray(data)

        oriImg.paste(captcha_image,box)
        composeGasmeterImg = np.array(oriImg)
        return captcha_text, composeGasmeterImg

    def get_compose_gasmeter_next_batch(self, gasmeterImgObj, batchsize = None):
        """
        :TODO

        根据燃气表图片文件名获得一批训练数据
        1 合成燃气表图片
        2 对合成图片进行处理，获得数字区域数据
        3 对数字区域数据进行大小调整，默认调整到 64 * 128
        4 重复 1/2/3步骤，生成一个批次
        :param gasmeterImgObj:cv2 读取的燃气表图片对象
        :param batchsize:
        :return:
        """
        if batchsize is None:
            batchsize = 64
        batch_x = np.zeros([batchsize, self._picBoxHeight * self._picBoxWidth * 3],dtype=np.float32)
        CHAR_SET_LEN = len(self._charset) + 1
        batch_y = np.zeros([batchsize, self._picCharacterLength * CHAR_SET_LEN],dtype=np.float32)

        def wrap_gen_text_and_image(gasmeterImgObj):
            """
            将合成图片进行处理，只保留取得的表头数字区域数据，并调整大小到 self._picBoxHeight * self._picBoxWidth
            将文本和表头数字区图片数据返回
            :param gasmeterImgObj:
            """
            text,image = self.get_compose_gasmeter_text_and_image(gasmeterImgObj)
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            image = ImageTool.getGasmeterAreaData(image)
            image = cv2.resize(image,(self._picBoxWidth,self._picBoxHeight))
            image = np.array(image,dtype=np.float32)

            # 显示截取的燃气表数字区的图片和文字
            # plt.figure()
            # plt.title(text)
            # plt.imshow(image)
            # plt.show()

            return text,image

        for i in range(batchsize):
            text, image = wrap_gen_text_and_image(gasmeterImgObj)

            batch_x[i, :] = image.flatten()
            batch_y[i, :] = self._text2vec(text)

        # 返回该训练批次
        return batch_x, batch_y


    def get_text_and_image(self):
        """ 生成字符序列和对应的图片数据 """
        image = ImageCaptcha(width=self._picBoxWidth, height=self._picBoxHeight,
                             backgroundColor=self._backgroundColor, fontColor=self._fontColor)

        captcha_text = self._random_text()
        captcha_text = ''.join(captcha_text)

        captcha = image.generate(captcha_text)

        captcha_image = Image.open(captcha)
        captcha_image = np.array(captcha_image)
        return captcha_text, captcha_image

    def get_next_batch(self, batchsize = None):
        """
        生成一个训练 batch
        """
        if batchsize is None:
            batchsize = 64
        batch_x = np.zeros([batchsize, self._picBoxHeight * self._picBoxWidth * 3])
        CHAR_SET_LEN = len(self._charset) + 1
        batch_y = np.zeros([batchsize, self._picCharacterLength * CHAR_SET_LEN])

        # 有时生成图像大小不是(60, 160, 3)
        def wrap_gen_captcha_text_and_image():
            ''' 获取一张图，判断其是否符合（60，160，3）的规格'''
            while True:
                text, image = self.get_text_and_image()
                if image.shape == (self._picBoxHeight, self._picBoxWidth, 3):  # 此部分应该与开头部分图片宽高吻合
                    return text, image

        for i in range(batchsize):
            text, image = wrap_gen_captcha_text_and_image()

            # 将图片数组一维化 同时将文本也对应在两个二维组的同一行
            batch_x[i, :] = image.flatten()
            batch_y[i, :] = self._text2vec(text)
        # 返回该训练批次
        return batch_x, batch_y

    def _text2vec(self,text):
        """	将验证码字符串转换成对应编码矢量	"""
        text_len = len(text)
        if text_len > self._picCharacterLength:
            raise ValueError('max length of captcha character is %d'%self._picCharacterLength)

        # CHAR_SET_LEN = self._charset + '_'，'_'表示字符为空，有时候识别的字符串长度不够指定的长度，用'_' 代替空字符串
        CHAR_SET_LEN = len(self._charset) + 1
        vector = np.zeros(self._picCharacterLength * CHAR_SET_LEN)

        def char2pos(c):
            if c == '_':
                k = 10
                return k
            if c >= '0' and c <= '9':
                k = ord(c) - ord('0')
            else:
                raise Exception('character not in 0-9')
            return k

        for i, c in enumerate(text):
            idx = i * CHAR_SET_LEN + char2pos(c)
            vector[idx] = 1
        return vector

    def _vec2text(self,vec):
        """
        向量转回文本
        将矢量转换成对应字符串
        """
        char_pos = vec.nonzero()[0]
        text = []
        # CHAR_SET_LEN = self._charset + '_'，'_'表示字符为空，有时候识别的字符串长度不够指定的长度，用'_' 代替空字符串
        CHAR_SET_LEN = len(self._charset) + 1
        for i, c in enumerate(char_pos):
            char_idx = c % CHAR_SET_LEN
            if char_idx < 10:
                char_code = char_idx + ord('0')
            elif char_idx == 10:
                char_code = ord('_')
            else:
                raise ValueError('error')
            text.append(chr(char_code))
        return "".join(text)

class ImageTool():
    """
    showGasmeterArea(filename):显示文件中燃气表数字所在区域。说明：燃气表数字应在图片居中位置，表头大小 <= 图片总大小的80%
    图片背景颜色最好是亮色系，不能是黑色。最好是白色。
    """

    @staticmethod
    def convertCv2ColorImg2Gray(bgrimg):
        """
        将cv2 读进来的图像转换成灰度图像对象并返回
        :param bgrimg:
        """
        return cv2.cvtColor(bgrimg,cv2.COLOR_BGR2GRAY)

    @staticmethod
    def preProcessImage(img, width=None):
        """
        将输入图片缩放到指定大小，默认是800*800。裁剪所得图片中间80%，并恢复到指定大小
        :param img: 输入图片数据，cv2读取的 object 对象
        :param width: 指定图片的大小
        """
        if width is None:
            width = 800
        img = cv2.resize(img, (width, width), interpolation=cv2.INTER_LINEAR)

        boxhw = int(width * 0.8)
        starthw = int((width - boxhw) / 2)
        # cropImg = image[y1:y1+hight, x1:x1+width]图片裁剪
        img = img[starthw:starthw + boxhw, starthw:starthw + boxhw]

        img = cv2.resize(img, (width, width), interpolation=cv2.INTER_LINEAR)
        return img


    @staticmethod
    def showGasmeterArea(imgobj,filename = None):
        """
        显示获得的图片区域，自己对原始图片进行了处理，看到的图片大小跟原始图片不一样
        :param imgobj:通过cv2 读进来的图像对象
        :param filename: 要处理的原始图片

        """

        imgTemp = imgobj.copy()
        imgTemp = cv2.cvtColor(imgTemp,cv2.COLOR_BGR2GRAY)

        box = ImageTool.getGasmeterRectBoxCornerPoint(grayImg=imgTemp)
        cimgobj = imgobj.copy()
        cimgobj = ImageTool.preProcessImage(cimgobj)
        cv2.drawContours(cimgobj, [box], 0, (0, 255, 0), 2)
        cv2.imshow("box area, %s"%FileNameUtil.getFilenameFromFullFilepathname(filename), cimgobj)

        k = cv2.waitKey(0)  # 无限期等待输入
        if k == 27:  # 如果输入ESC退出
            cv2.destroyAllWindows()

    @staticmethod
    def getGasmeterAreaData(imgobj):
        """
        根据图片文件获得燃气表数字所在区域的图片数据并返回。
        :param imgobj: 燃气表图片，通过cv2读进来的图片数据对象
        """
        imgTemp = imgobj.copy()
        imgTemp = cv2.cvtColor(imgTemp, cv2.COLOR_BGR2GRAY)
        box = ImageTool.getGasmeterRectBoxCornerPoint(grayImg=imgTemp)

        cimgobj = imgobj.copy()
        cimgobj = ImageTool.preProcessImage(cimgobj)
        return ImageTool._getCropImage(cimgobj, box)

    # @staticmethod
    # def getGasmeterCompositePicture(filename, captcha):
    #     """
    #     将 captcha Image对象合成到燃气表的数字区域，返回合成的图片 PIL.Image 对象
    #     :param filename:燃气表图片文件名
    #     :param captcha:要向燃气表数字区黏贴的图片，类型是PIL.Image对象
    #     :return:
    #     """
    #     boxCornerPoint = ImageTool.getGasmeterRectBoxCornerPoint(filename)
    #     box = ImageTool.getBoxFromBoxCorner(boxCornerPoint)
    #     box = (box[0],box[1],box[0] + captcha.width,box[1] + captcha.height)
    #     img = cv2.imread(filename)
    #     img = ImageTool.preProcessImage(img)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     data = np.array(img)
    #     img = Image.fromarray(data)
    #
    #     img.paste(captcha, box)
    #     return img


    @staticmethod
    def getGasmeterRectBoxCornerPoint(filename =None, grayImg = None):
        """
        获得燃气表数字区域，返回的是区域的四个顶点，优先使用filename 文件名
        :param filename:想要获取表头四角的图片文件名
        :param grayImg:通过 cv2 读进来的灰度图片对象
        :return:
        """
        if grayImg is not None:
            grayImg = grayImg.copy()
        if filename is not None:
            grayImg = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        grayImg = ImageTool.preProcessImage(grayImg)
        # 高斯滤波，去除背景噪声
        grayImg = cv2.GaussianBlur(grayImg, (7, 7), 0)
        # 进行二值化处理
        _, binImg = cv2.threshold(grayImg, 50, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closedImg = cv2.morphologyEx(binImg, cv2.MORPH_CLOSE, kernel)
        openedImg = cv2.morphologyEx(closedImg, cv2.MORPH_OPEN, kernel)

        # cv2.erode 腐蚀图像
        # eroded = cv2.erode(img, kernel);
        # cv2.dilate 膨胀图像
        # dilated = cv2.dilate(img, kernel)

        erodeImg = openedImg
        erokernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilkernal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        for i in range(2):
            erodeImg = cv2.dilate(erodeImg, dilkernal)
            erodeImg = cv2.erode(erodeImg, erokernel)

        _, erodeImg = cv2.threshold(erodeImg, 200, 255, cv2.THRESH_BINARY)

        # 找出区域的轮廓。cv2.findContours()函数
        # (cnts, _) = cv2.findContours(erodeImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if "Linux" in platform.system():
            cnts, hierarchy = cv2.findContours(erodeImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            (erodeImg, cnts, hierarchy) = cv2.findContours(erodeImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        return (x1,y1,x2,y2)

    @staticmethod
    def getBoxWidth(box):
        return box[2]-box[0]

    @staticmethod
    def getBoxHeight(box):
        return box[3]-box[1]


    @staticmethod
    def _getCropImage(image, boxCornerPoint):
        """
        获取裁剪图片
        :param image 是cv2读进来的图片对象
        :param boxCornerPoint: 要裁剪的矩形四个顶角坐标
        """
        box = ImageTool.getBoxFromBoxCorner(boxCornerPoint)
        hight = ImageTool.getBoxHeight(box)
        width = ImageTool.getBoxWidth(box)
        left = box[0]
        upper = box[1]

        cropImg = image[upper:upper + hight, left:left + width]
        return cropImg


def testShowGasmeterArea():
    imgdirname = ["data","img"]
    imgdirname = FileNameUtil.getDirname(FileNameUtil.getBasedirname(__file__),imgdirname)
    pattern = r'.*\.jpg$'
    filelist = FileNameUtil.getPathFilenameList(imgdirname,pattern)
    for each in filelist:
        ImageTool.showGasmeterArea(cv2.imread(each),filename=each)

def testgetGasmeterAreaData():
    imgdirname = ["data", "img"]
    imgdirname = FileNameUtil.getDirname(FileNameUtil.getBasedirname(__file__), imgdirname)
    pattern = r'.*\.jpg$'
    filelist = FileNameUtil.getPathFilenameList(imgdirname, pattern)
    for each in filelist:
        img = ImageTool.getGasmeterAreaData(cv2.imread(each))
        data = np.array(img)
        print(data.shape)
        plt.figure()
        plt.title("gasmeter area")
        plt.imshow(img)
        plt.show()

def testGasmeterComposite():
    captchaCharacterLength = 5
    captchaBoxWidth = 128
    captchaBoxHeight = 64
    gen = GenDigitPicture(captchaCharacterLength, captchaBoxWidth, captchaBoxHeight,
                          backgroundColor=(1,1,1),fontColor=(200,200,200))

    imgdirname = ["data", "img"]
    imgdirname = FileNameUtil.getDirname(FileNameUtil.getBasedirname(__file__), imgdirname)
    pattern = r'.*\.jpg$'
    filename = FileNameUtil.getPathFilenameList(imgdirname, pattern)[0]
    gasmeterPic = cv2.imread(filename)
    while (1):
        text, image = gen.get_compose_gasmeter_text_and_image(gasmeterPic)
        print('begin ' + time.strftime("%Y-%m-%d %H:%M:%S") + str(type(image)))
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
        plt.imshow(image)

        plt.show()
        print('end ' + time.strftime("%Y-%m-%d %H:%M:%S"))



def testCaptchaGenerate():
    # 验证码一般都无视大小写；验证码长度4个字符
    captchaCharacterLength = 5
    captchaBoxWidth = 128
    captchaBoxHeight = 64
    gen = GenDigitPicture(captchaCharacterLength, captchaBoxWidth, captchaBoxHeight)
    while (1):
        text, image = gen.get_text_and_image()

        print('begin ' + time.strftime("%Y-%m-%d %H:%M:%S") + str(type(image)))
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
        plt.imshow(image)

        plt.show()
        print('end ' + time.strftime("%Y-%m-%d %H:%M:%S"))

def testGet_compose_gasmeter_next_batch():
    imgdirname = ["data", "img"]
    imgdirname = FileNameUtil.getDirname(FileNameUtil.getBasedirname(__file__), imgdirname)
    pattern = r'.*\.jpg$'
    filelist = FileNameUtil.getPathFilenameList(imgdirname, pattern)

    captchaCharacterLength = 5
    captchaBoxWidth = 128
    captchaBoxHeight = 64
    gen = GenDigitPicture(captchaCharacterLength, captchaBoxWidth, captchaBoxHeight,
                          backgroundColor=(10, 10, 10), fontColor=(200, 200, 200))

    for eachfile in filelist:
        gen.get_compose_gasmeter_next_batch(cv2.imread(eachfile))





def test():
    testShowGasmeterArea()
    # testCaptchaGenerate()
    # testgetGasmeterAreaData()
    # testGasmeterComposite()
    # testGet_compose_gasmeter_next_batch()

if __name__ == '__main__':
    test()

