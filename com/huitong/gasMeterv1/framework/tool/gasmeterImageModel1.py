#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/26.
"""
from com.huitong.gasMeterv1.framework.tool.GenDigitsImage import GenDigitsPicture
from com.huitong.gasMeterv1.framework.tool.ImageTool import ImageTool
from com.huitong.gasMeterv1.framework.tool.CaptchaTool import ImageCaptcha
from com.huitong.gasMeterv1.framework.tool.filenameUtil import FileNameUtil
from PIL import Image

import numpy as np
import random





class GenImageGasMeterStyle1m1(GenDigitsPicture):
    """
    GenImageGasMeterStyle1m1:GenImageGasMeterStyle1 model1
    """

    def get_text_and_image(self):
        """ 生成字符序列和对应的图片数据 图片颜色通道是(R,G,B)-> text,image"""
        charBoxWidth = 20
        charBoxHeight = 50
        blackBkgColor = ImageCaptcha.random_bkg_color(0,65)
        fontColor = ImageCaptcha.random_font_color(180,250)
        grayBkgColor = ImageCaptcha.random_bkg_color(130,220)

        image = ImageCaptcha(width=charBoxWidth, height=charBoxHeight,
                             backgroundColor=blackBkgColor, fontColor=fontColor)

        captcha_text = self._random_text()
        captcha_text = ''.join(captcha_text)

        captcha_image = Image.new('RGB',(self._picBoxWidth,self._picBoxHeight),grayBkgColor)
        dx = 4
        dy = 7
        desX = dx

        for c in captcha_text:
            charImage = image.generate(c)
            charImage = Image.open(charImage)
            captcha_image.paste(charImage,(desX,dy))
            desX += dx + charBoxWidth



        if self._imageDepth == 1:
            # 将彩色图片转换成灰度图图片
            captcha_image = ImageTool.convertImgRGB2Gray(captcha_image)

        captcha_image = np.array(captcha_image)
        return captcha_text, captcha_image





class GenImageGasMeterStyle1m2(GenDigitsPicture):
    """
    GenImageGasMeterStyle1m2:GenImageGasMeterStyle1 model2
    使用目录 img/gasmeterRoller 下的图片生成训练模板进行训练
    """

    @staticmethod
    def getRandomFilename():
        dirname = r"D:\chengxu\python\project\digitRecognise\com\huitong\gasMeterv1\data\img\gasmeterRoller"
        filenameList = FileNameUtil.getPathFilenameList(dirname)
        return random.choice(filenameList)




    def get_text_and_image(self,backgroundColor = None, fontColor = None, fontSizes = (28,)):
        """ 生成字符序列和对应的图片数据 图片颜色通道是(R,G,B)-> text,image"""
        charBoxWidth = 15
        charBoxHeight = 30
        blackBkgColor = ImageCaptcha.random_bkg_color(0,65)
        fontColor = ImageCaptcha.random_font_color(180,250)
        grayBkgColor = ImageCaptcha.random_bkg_color(130,220)

        image = ImageCaptcha(width=charBoxWidth, height=charBoxHeight,
                             backgroundColor=blackBkgColor, fontColor=fontColor, font_sizes=fontSizes)

        captcha_text = self._random_text()
        captcha_text = ''.join(captcha_text)

        filename = GenImageGasMeterStyle1m2.getRandomFilename()
        captcha_image = Image.open(filename)
        captcha_image = captcha_image.resize((self._picBoxWidth, self._picBoxHeight),Image.CUBIC)
        dxs = [7,32,58,83,107]
        dy = 19
        i=0

        for c in captcha_text:
            desX = dxs[i]
            charImage = image.generate(c)
            charImage = Image.open(charImage)
            captcha_image.paste(charImage,(desX,dy))
            i +=1



        if self._imageDepth == 1:
            # 将彩色图片转换成灰度图图片
            captcha_image = ImageTool.convertImgRGB2Gray(captcha_image)

        captcha_image = np.array(captcha_image)
        return captcha_text, captcha_image

def test():
    import time
    import matplotlib.pyplot as plt

    captchaCharacterLength = 5
    captchaBoxWidth = 128
    captchaBoxHeight = 64
    gen = GenImageGasMeterStyle1m2(captchaCharacterLength, captchaBoxWidth, captchaBoxHeight, imageDepth=1)

    while (1):
        text, image = gen.get_text_and_image()

        print('begin ' + time.strftime("%Y-%m-%d %H:%M:%S") + str(type(image)))
        plt.figure()
        plt.imshow(image)
        plt.title(text + str(image.shape))

        plt.show()
        print('end ' + time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    test()