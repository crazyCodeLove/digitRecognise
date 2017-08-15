#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/12.
"""
from multiprocessing import Process

from com.huitong.gasMeterv1.genDigitPic import GenDigitPicture
from collections import namedtuple
from com.huitong.gasMeterv1 import ModelUtil
from com.huitong.gasMeterv1 import ResNetModel
from com.huitong.gasMeterv1.filenameUtil import FileNameUtil

import tensorflow as tf
import os
import pickle
import random
import cv2



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

logger = None
save_file_name = None

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


def startTrain(trainepochnums,
               hps,
               mode,
               gps,
               save_file_name):
    logger = ModelUtil.MyLog(getFilename(gps.logFilename, dirnameList=gps.logDirnameList))
    xp = tf.placeholder(tf.float32, [None, captchaBoxHeight * captchaBoxWidth * gen.ImageDepth])
    yp = tf.placeholder(tf.float32, [None, captchaCharacterLength*CHAR_SET_LEN])
    model = ResNetModel.ResNetModel(hps, xp, yp, mode, captchaBoxHeight, captchaBoxWidth, gen.ImageDepth)
    model.create_graph(captchaCharacterLength)

    gasmeterPictureFilenameList = FileNameUtil.getDirname(FileNameUtil.getBasedirname(__file__),["data","img"])
    gasmeterPictureFilenameList = FileNameUtil.getPathFilenameList(gasmeterPictureFilenameList, r".*\.jpg")

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver()
        with open(gps.peizhi_filename, mode='rb') as rfobj:
            peizhi = pickle.load(rfobj)

        if not peizhi['is_restore']:
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess, save_file_name)

        base_step = peizhi['train_step']
        end_step = int(base_step + 50000*trainepochnums / hps.batch_nums +1)
        tpic = random.choice(gasmeterPictureFilenameList)
        img = cv2.imread(tpic)

        for itstep in range(base_step,end_step):
            # images,labels = gen.get_next_batch(hps.batch_nums)

            images, labels = gen.get_compose_gasmeter_next_batch(img, batchsize=hps.batch_nums)

            feed_dict = {
                xp:images,
                yp:labels,
                model.learning_rate: peizhi['lrn_rate'],
                model.is_training_ph: True}
            (inlabels,outputs,cost,_ ) = sess.run(
                [model.labes, model.outputs, model.loss, model.train_op],
                feed_dict=feed_dict)

            if itstep % 50 == 0:
                trainacc = ModelUtil.get_str_accurate(outputs,inlabels,captchaCharacterLength,CHAR_SET_LEN)
                msg = "trainstep:%5d  loss:%e  train acc:%.5f"%(itstep,cost,trainacc)

                if itstep % 200 ==0:
                    logger.showAndLogMsg(msg)
                else:
                    logger.log_message(msg)

            # if itstep % 20 ==0 and itstep > 0:
            #     print("before save")
            #     saver.save(sess=sess, save_path=save_file_name)
            #     print("after save")

        print("before save")
        saver.save(sess=sess,save_path=save_file_name)
        print("after save")
        ModelUtil.update_peizhi(gps.peizhi_filename,'is_restore',True)
        ModelUtil.update_peizhi(gps.peizhi_filename,'train_step',end_step)



def train_main():
    global save_file_name,logger
    hps = HParams(batch_nums=64,
                  num_classes=10,
                  deep_net_fkn=30,
                  img_depth=gen.ImageDepth,
                  deepk=[2,1.8,1.5],
                  carriage_block_num=[2,2,2])



    gps = GParams(
        peizhi_filename="peizhi.xml",
        saveVariableDirnameList=["data","digitRecognise","temp"],
        saveVariableFilename= "temp.ckpy",
        logDirnameList=["data", "log"],
        logFilename = "resnetGasmeterv1log1.txt")

    save_file_name = getFilename(gps.saveVariableFilename, dirnameList=gps.saveVariableDirnameList)

    def dataDirInit():
        path = FileNameUtil.getBasedirname(__file__)
        saveVariableDirname = FileNameUtil.getDirname(path, gps.saveVariableDirnameList)
        logDirname = FileNameUtil.getDirname(path, gps.logDirnameList)
        if not FileNameUtil.fileExisted(saveVariableDirname):
            os.makedirs(saveVariableDirname)
        if not FileNameUtil.fileExisted(logDirname):
            os.makedirs(logDirname)

    dataDirInit()

    logger = ModelUtil.MyLog(getFilename(gps.logFilename,dirnameList=gps.logDirnameList))

    ModelUtil.init_peizhi(
        peizhifilename=gps.peizhi_filename, peizhidict=peizhi_dict)

    msg = "peizhi\nhps:"+str(hps) +\
          "\ngps:"+str(gps)+ "\n deepk and descrate can be fractional, " \
          "captchaBoxHeight=%d,captchaBoxWidth=%d,captchaCharacterLength=%d"%(captchaBoxHeight,captchaBoxWidth,captchaCharacterLength)
    logger.showAndLogMsg(msg)

    while True:
        print("start training")
        mode = 'train'
        trainNumsBeforeValid = 6
        p = Process(target=startTrain,args=(trainNumsBeforeValid, hps, mode, gps, save_file_name))
        p.start()
        p.join()
        print("training end")

        with open(gps.peizhi_filename,mode='rb') as rfobj:
            peizhi = pickle.load(rfobj)
        if peizhi['max_test_acc'] >= 0.999:
            print("already over best test acc, now test acc is %.4f"% peizhi['max_test_acc'])
            break



if __name__ == "__main__":
    train_main()

