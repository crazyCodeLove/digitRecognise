#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/9.
"""

from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random,time,os

# 验证码中的字符, 就不用汉字了
number = ['0','1','2','3','4','5','6','7','8','9']
# 验证码一般都无视大小写；验证码长度4个字符
captchaCharacterLength = 4
captchaBoxWidth = 160
captchaBoxHeight = 60



def random_captcha_text(char_set=number, captcha_size=captchaCharacterLength):
	captcha_text = []
	for i in range(captcha_size):
		c = random.choice(char_set)
		captcha_text.append(c)
	return captcha_text

# 生成字符对应的验证码
def gen_captcha_text_and_image():
	"""
	生成字符序列和对应的图片数据
	:return:
	"""
	image = ImageCaptcha(width=captchaBoxWidth, height=captchaBoxHeight)

	captcha_text = random_captcha_text()
	captcha_text = ''.join(captcha_text)

	captcha = image.generate(captcha_text)

	captcha_image = Image.open(captcha)
	captcha_image = np.array(captcha_image)
	return captcha_text, captcha_image

if __name__ == '__main__':
	# 测试
	while(1):
		text, image = gen_captcha_text_and_image()

		print(image.shape, "no chinese")
		print('%d, %.4f'% (15, 15.234587))

		print('begin ' + time.strftime("%Y-%m-%d %H:%M:%S") + str(type(image)))
		f = plt.figure()
		ax = f.add_subplot(111)
		ax.text(0.1, 0.9,text, ha='center', va='center', transform=ax.transAxes)
		plt.imshow(image)


		plt.show()
		print('end ' + time.strftime("%Y-%m-%d %H:%M:%S"))

