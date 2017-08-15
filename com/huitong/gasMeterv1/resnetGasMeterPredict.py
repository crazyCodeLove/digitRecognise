#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/15.
"""

from com.huitong.gasMeterv1.genDigitPic import GenDigitPicture
from collections import namedtuple
from com.huitong.gasMeterv1 import ResNetModel
from com.huitong.gasMeterv1.filenameUtil import FileNameUtil

import tensorflow as tf
import numpy as np


captchaCharacterLength = 5
captchaBoxWidth = 128
captchaBoxHeight = 64

gen = GenDigitPicture(captchaCharacterLength,captchaBoxWidth,captchaBoxHeight,
                      backgroundColor=(20, 20, 20), fontColor=(200, 200, 200))

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



def getPredict(hps, mode, gasmeter_filename, save_file_name):
    xp = tf.placeholder(tf.float32, [None, captchaBoxHeight * captchaBoxWidth * gen.ImageDepth])
    yp = tf.placeholder(tf.float32, [None, captchaCharacterLength * CHAR_SET_LEN])
    model = ResNetModel.ResNetModel(hps, xp, yp, mode, captchaBoxHeight, captchaBoxWidth, gen.ImageDepth)
    model.create_graph(captchaCharacterLength)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, save_file_name)

        images = gen.get_batch_gasmeter_digit_area_from_filename(gasmeter_filename, hps.batch_nums)
        feed_dict = {
            xp: images,
            model.is_training_ph: False}

        outputs = sess.run([model.outputs], feed_dict=feed_dict)
        text = get_predict_text(outputs)
    return text

def main(gasmeter_filename):
    hps = HParams(batch_nums=10,
                  num_classes=10,
                  deep_net_fkn=30,
                  img_depth=gen.ImageDepth,
                  deepk=[2, 1.8, 1.5],
                  carriage_block_num=[2, 2, 2])

    gps = GParams(
        peizhi_filename="peizhi.xml",
        saveVariableDirnameList=["data", "digitRecognise", "temp"],
        saveVariableFilename="temp.ckpy",
        logDirnameList=[""],
        logFilename="")

    save_file_name = getFilename(gps.saveVariableFilename, dirnameList=gps.saveVariableDirnameList)
    mode = "predict"
    return getPredict(hps, mode, gasmeter_filename, save_file_name)


if __name__ == "__main__":
    filename = r"D:\chengxu\python\project\digitRecognise\com\huitong\gasMeterv1\data\img\8.jpg"
    predict = main(filename)

    print("predict:%s"%predict)







