#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/16.
"""
import random
import numpy as np
from PIL import Image

from com.huitong.gasMeterv1.framework.tool.CaptchaTool import ImageCaptcha
from com.huitong.gasMeterv1.framework.tool.ImageTool import ImageTool

class GenDigitsPicture():
    """
    用于产生数字验证码、数字表头图片、模拟摄像头采集的图片
    :param captchaCharacterLength:验证码上字符个数
    :param captchaBoxWidth：验证码图片宽度
    :param captchaBoxHeight：验证码图片高度
    :param batchSize：生成一批次数量，默认值是 64
    :param charset：生成验证码的字符集，默认是['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    """


    def __init__(self, captchaCharacterLength, captchaBoxWidth, captchaBoxHeight,
                 charset = None, imageDepth=3):

        self._picCharacterLength = captchaCharacterLength
        self._picBoxWidth = captchaBoxWidth
        self._picBoxHeight = captchaBoxHeight

        if charset is None:
            self._charset = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        else:
            self._charset = charset

        self._imageDepth = imageDepth



    @property
    def ImageHeight(self):
        return self._picBoxHeight

    @property
    def ImageWidth(self):
        return self._picBoxWidth

    @property
    def ImageDepth(self):
        return self._imageDepth

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

    def get_text_and_image(self,backgroundColor = None, fontColor = None, fontSizes = None):
        """ 生成字符序列和对应的图片数据 图片颜色通道是(R,G,B)-> text,image"""

        image = ImageCaptcha(width=self._picBoxWidth, height=self._picBoxHeight,
                             backgroundColor=backgroundColor, fontColor=fontColor, font_sizes=fontSizes)

        captcha_text = self._random_text()
        captcha_text = ''.join(captcha_text)

        captcha = image.generate(captcha_text)

        captcha_image = Image.open(captcha)

        if self._imageDepth == 1:
            # 将彩色图片转换成灰度图图片
            captcha_image = ImageTool.convertImgRGB2Gray(captcha_image)

        captcha_image = np.array(captcha_image)
        return captcha_text, captcha_image

    def get_next_batch(self, batchsize = None, fontSizes = None):
        """
        生成一个训练 batch
        """
        if batchsize is None:
            batchsize = 64
        batch_x = np.zeros([batchsize, self._picBoxHeight * self._picBoxWidth * self._imageDepth])
        CHAR_SET_LEN = len(self._charset) + 1
        batch_y = np.zeros([batchsize, self._picCharacterLength * CHAR_SET_LEN])

        # 有时生成图像大小不是(60, 160, 3)
        def wrap_gen_captcha_text_and_image():
            ''' 获取一张图，判断其是否符合（60，160，depth）的规格'''
            while True:
                text, image = self.get_text_and_image(fontSizes=fontSizes)
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

    def vec2text(self, vec):
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

def testGenDigitsPicture():
    import time
    import matplotlib.pyplot as plt
    import cv2

    captchaCharacterLength = 5
    captchaBoxWidth = 128
    captchaBoxHeight = 64
    gen = GenDigitsPicture(captchaCharacterLength, captchaBoxWidth, captchaBoxHeight,imageDepth=1)
    while (1):
        text, image = gen.get_text_and_image()

        print('begin ' + time.strftime("%Y-%m-%d %H:%M:%S") + str(type(image)))
        plt.figure()
        plt.imshow(image)
        plt.title(text + str(image.shape))

        plt.show()
        print('end ' + time.strftime("%Y-%m-%d %H:%M:%S"))

        # image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        # cv2.imshow("gray",image)
        # k = cv2.waitKey(0)
        # if k == 27:
        #     cv2.destroyAllWindows()














if __name__ == "__main__":
    testGenDigitsPicture()
