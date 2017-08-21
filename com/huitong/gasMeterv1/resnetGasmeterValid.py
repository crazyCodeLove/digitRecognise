#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/21.
"""


from collections import namedtuple

import numpy as np
import tensorflow as tf
import cv2
import platform

from com.huitong.gasMeterv1 import ResNetModel
from com.huitong.gasMeterv1.framework.tool.filenameUtil import FileNameUtil
from com.huitong.gasMeterv1.framework.tool.GenDigitsImage import GenDigitsPicture
from com.huitong.gasMeterv1.framework.gasmeterModel.gasmeterStyle0 import GasmeterStyle0
from com.huitong.gasMeterv1.framework.gasmeterModel.gasmeterStyle1 import GasmeterStyle1
from com.huitong.gasMeterv1.framework.tool.ImageTool import ImageTool

captchaCharacterLength = 5
captchaBoxWidth = 128
captchaBoxHeight = 64

gen = GenDigitsPicture(captchaCharacterLength,captchaBoxWidth,captchaBoxHeight,
                       imageDepth=1)

CHAR_SET_LEN = len(gen.CharSet) + 1   # 字符集中字符数量

HParams = namedtuple('HParams',
                     'batch_nums, num_classes, deep_net_fkn,'
                     'img_depth, deepk, carriage_block_num')

GParams = namedtuple('GParams',
                     'peizhi_filename,saveVariableDirnameList,saveVariableFilename,'
                     'logDirnameList,logFilename')

peizhi_dict = {'lrn_rate':1e-2,
               'is_restore':False,
               'train_step':0,
               'test_step':0,
               'max_test_acc':0}


def getFilename(filename, baseDirname = None, dirnameList = None):
    """
    根据基目录、子目录名列表、文件名获取文件全路径名。
    如果基目录名为None,则基目录为当前文件所在目录。
    :param dirnameList:
    :param filename:
    :param baseDirname:
    """
    path = baseDirname
    if baseDirname is None:
        path = FileNameUtil.getBasedirname(__file__)

    if dirnameList is not None:
        path = FileNameUtil.getDirname(path,dirnameList)
    return FileNameUtil.getFilename(path, filename)

def get_predict_text(outputs):
    predict = np.reshape(outputs, [-1, captchaCharacterLength, CHAR_SET_LEN])
    max_idx_p = np.argmax(predict, 2)

    predict = max_idx_p[0]
    vector = np.zeros(captchaCharacterLength*CHAR_SET_LEN)
    i = 0
    for index in predict:
        vector[CHAR_SET_LEN*i + index] = 1
        i += 1

    return gen.vec2text(vector)

def get_most_common_digit_in_column(data):
    """
    找出二维数组中每列出现频次最高的数字
    :param data: numpy 2D 数组
    """
    shape = data.shape




def getPredict(hps, mode, save_file_name):
    xp = tf.placeholder(tf.float32, [None, captchaBoxHeight * captchaBoxWidth * gen.ImageDepth])
    yp = tf.placeholder(tf.float32, [None, captchaCharacterLength * CHAR_SET_LEN])
    model = ResNetModel.ResNetModel(hps, xp, yp, mode, captchaBoxHeight, captchaBoxWidth, gen.ImageDepth)
    model.create_graph(captchaCharacterLength)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, save_file_name)

        while True:
            oriText,image = gen.get_text_and_image()
            images = ImageTool.repeatImage2Tensor(image,hps.batch_nums)


            feed_dict = {
                xp: images,
                model.is_training_ph: False}

            outputs = sess.run([model.outputs], feed_dict=feed_dict)
            predictText = get_predict_text(outputs)

            title = "text:%s, predict:%s"%(oriText,predictText)

            ImageTool.showImagePIL(image, title)
            print("predict:%s"%predictText)

def main():
    hps = HParams(batch_nums=10,
                  num_classes=10,
                  deep_net_fkn=30,
                  img_depth=gen.ImageDepth,
                  deepk=[2, 1.8, 1.8],
                  carriage_block_num=[2, 2, 2])

    gps = GParams(
        peizhi_filename="peizhi.xml",
        saveVariableDirnameList=["data", "digitRecognise", "temp"],
        saveVariableFilename="temp.ckpy",
        logDirnameList=[""],
        logFilename="")

    save_file_name = getFilename(gps.saveVariableFilename, dirnameList=gps.saveVariableDirnameList)
    mode = "predict"
    getPredict(hps, mode, save_file_name)


if __name__ == "__main__":

    main()

