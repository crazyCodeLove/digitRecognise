#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/11.
"""


import os
import random
from PIL import Image
from PIL import ImageFilter
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

def getFounts():
    path = os.path.dirname(__file__)
    for i in range(2):
        path = os.path.dirname(path)

    path = os.path.join(path,'data')
    path = os.path.join(path,'fonts')
    DEFAULT_FONTS = [os.path.join(path, font) for font in os.listdir(path)]
    return DEFAULT_FONTS




DEFAULT_FONTS = getFounts()
DEFAULT_FOUNT_SIZES = [i for i in range(29, 38, 2)]

__all__ = ['ImageCaptcha']


table  =  []
for  i  in  range( 256 ):
    table.append( i * 1.97 )


class _Captcha(object):
    def generate(self, chars, format='png'):
        """Generate an Image Captcha of the given characters.

        :param chars: text to be generated.
        :param format: image file format
        """
        im = self.generate_image(chars)
        out = BytesIO()
        im.save(out, format=format)
        out.seek(0)
        return out

    def write(self, chars, output, format='png'):
        """Generate and write an image CAPTCHA data to the output.

        :param chars: text to be generated.
        :param output: output destionation.
        :param format: image file format
        """
        im = self.generate_image(chars)
        return im.save(output, format=format)


class ImageCaptcha(_Captcha):
    """Create an image CAPTCHA.

    Many of the codes are borrowed from wheezy.captcha, with a modification
    for memory and developer friendly.

    ImageCaptcha has one built-in font, DroidSansMono, which is licensed under
    Apache License 2. You should always use your own fonts::

        captcha = ImageCaptcha(fonts=['/path/to/A.ttf', '/path/to/B.ttf'])

    You can put as many fonts as you like. But be aware of your memory, all of
    the fonts are loaded into your memory, so keep them a lot, but not too
    many.

    :param width: The width of the CAPTCHA image.
    :param height: The height of the CAPTCHA image.
    :param fonts: Fonts to be used to generate CAPTCHA images.
    :param font_sizes: Random choose a font size from this parameters.
    :param backgroundColor: assign a background color,example:(100,100,100) is RGB(100,100,100)
    :param fontColor: assign a fount color, format like backgroundColor
    """
    def __init__(self, width=160, height=60, fonts=None, font_sizes=None,
                 backgroundColor = None, fontColor = None):
        self._width = width
        self._height = height
        self._fonts = fonts or DEFAULT_FONTS
        self._font_sizes = font_sizes or DEFAULT_FOUNT_SIZES
        self._truefonts = []
        self._backgroundColor = backgroundColor
        self._fontColor = fontColor

    @property
    def truefonts(self):
        if self._truefonts:
            return self._truefonts
        self._truefonts = tuple([
            truetype(n, s)
            for n in self._fonts
            for s in self._font_sizes
        ])
        return self._truefonts

    @staticmethod
    def create_noise_curve(image, color):
        w, h = image.size
        x1 = random.randint(0, int(w / 5))
        x2 = random.randint(w - int(w / 5), w)
        y1 = random.randint(int(h / 5), h - int(h / 5))
        y2 = random.randint(y1, h - int(h / 5))
        points = [x1, y1, x2, y2]
        end = random.randint(160, 200)
        start = random.randint(0, 20)
        Draw(image).arc(points, start, end, fill=color)
        return image

    @staticmethod
    def create_noise_dots(image, color, width=2, number=1):
        draw = Draw(image)
        w, h = image.size
        while number:
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=width)
            number -= 1
        return image


    def random_height_stretch(self, image, maxval = None):
        """对上下方向上进行随机拉伸，然后裁剪到原始大小"""
        if maxval is None:
            maxval = 30
        addHeight = random.randint(1,maxval)
        midAddHeight = int(addHeight/2)
        image = image.resize((self._width,self._height + addHeight))

        # box变量是一个四元组(左，上，右，下)。
        box = (0, midAddHeight, self._width, self._height + midAddHeight)
        image = image.crop(box)
        return image




    def create_captcha_image(self, chars, color, background):
        """Create the CAPTCHA image itself.

        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.

        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        """
        image = Image.new('RGB', (self._width, self._height), background)
        draw = Draw(image)

        def _draw_character(c):
            font = random.choice(self.truefonts)
            w, h = draw.textsize(c, font=font)

            dx = random.randint(0, 4)
            dy = random.randint(0, 4)
            im = Image.new('RGBA', (w + dx, h + dy))
            Draw(im).text((dx, dy), c, font=font, fill=color)

            # rotate
            im = im.crop(im.getbbox())
            im = im.rotate(random.uniform(-15, 15), Image.BILINEAR, expand=1)

            return im

        images = []
        for c in chars:
            images.append(_draw_character(c))

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        average = int(text_width / len(chars))
        rand = int(0.15 * average)
        offset = int(average * 0.15)

        for im in images:
            w, h = im.size
            mask = im.convert('L').point(table)
            image.paste(im, (offset, int((self._height - h) / 2)), mask)
            offset = offset + w + random.randint(2,2+rand)

        if width > self._width:
            image = image.resize((self._width, self._height))



        return image

    def generate_image(self, chars):
        """Generate the image of the given characters.

        :param chars: text to be generated.
        """
        #####*************************** background color and font color
        if self._backgroundColor:
            background = self._backgroundColor
        else:
            background = ImageCaptcha.random_bkg_color(0,65)

        if self._fontColor:
            color = self._fontColor
        else:
            color = ImageCaptcha.random_font_color(140, 250, random.randint(220, 255))

        #####***************************

        im = self.create_captcha_image(chars, color, background)
        self.create_noise_dots(im, ImageCaptcha.random_noise_color())
        # self.create_noise_curve(im, ImageCaptcha.random_noise_color())
        # im = self.random_height_stretch(im)
        im = im.filter(ImageFilter.SMOOTH)
        return im

    @staticmethod
    def random_font_color(start, end, opacity=None):
        red = random.randint(start, end)
        green = red + random.randint(0, 25)
        blue = red + random.randint(0, 25)
        if opacity is None:
            return (red, green, blue)
        return (red, green, blue, opacity)

    @staticmethod
    def random_bkg_color(start, end, opacity=None):
        red = random.randint(start, end)
        green = red + random.randint(0, 25)
        blue = red + random.randint(0, 25)
        if opacity is None:
            return (red, green, blue)
        return (red, green, blue, opacity)

    @staticmethod
    def random_noise_color():
        start = 10
        end = 250
        red = random.randint(start, end)
        green = random.randint(start, end)
        blue = random.randint(start, end)
        opacity = random.randint(start, end)

        return (red, green, blue, opacity)


