#!/usr/bin/env python
#coding=utf-8

import tensorflow as tf
from com.huitong.gasMeterv1 import ModelUtil


class ResNetModel(object):
    """
    :param hps: 表示超参数，含有如下参数
        hps.img_depth：输入图片深度,
        hps.deep_net_fkn: 第一次卷积后期望输出tensor 的 depth
        hps.batch_nums: 一次训练的样本数
        hps.carriage_block_num: list,example[2,2,2]:每层 carriage 数量
        hps.deepk: list,每次 pooling 后 depth 要变化的倍数,example,[2,2,2]
        hps.num_classes: 要分的种类数目




    :param images: 4D tensor,[batch,img_height,img_width,img_depth]
    :param labels: 2D tensor,[batch,digit_code_vector]
    :param mode: 是否是在训练，"train":表示在训练，"test":在测试
    """

    def __init__(self,hps,images,labels, mode, boxHeight, boxWidth, boxDepth):
        """
        进行初始化
        """
        self.hps = hps
        self._images = images
        self.labes = labels
        self.mode = mode
        self._boxHeight = boxHeight
        self._boxWidth = boxWidth
        self._boxDepth = boxDepth


    def create_deep_res_head(self, inputx, is_training_ph, activateFunc=tf.nn.relu):
        """
        会进行一次max pooling,
        一层卷积层,一层max pooling,一个building block serial组成
        :param inputx: inputs shape is 4D tensor, value is arrange(0,1.0),或者(0,255)
        """
        # kernal_width = 5
        kernal_width = 3
        depth = self.hps.deep_net_fkn

        inputx = tf.reshape(inputx, [-1, self._boxHeight, self._boxWidth, self._boxDepth])


        # conv_layer1 is 64*64*deep_net_fkn
        outputs = ModelUtil.add_BN_conv_layer(
            inputx, kernal_width, self.hps.img_depth,
            depth, is_training_ph, scope="reshead",
            activateFunc=activateFunc,stride=[1,1,1,1])


        return outputs

    def create_deep_res_body(
            self,batch_num, carriage_block_num, inputs, is_training_ph,
            layername="layer", activateFunc=tf.nn.relu, stride=[1, 1, 1, 1]):
        """

        最后的输出是4*4*deep_net_fkn*(2**4)
        """
        kernalWidth = 3

        outputs = inputs
        for it in range(len(carriage_block_num)):
            outputs = ModelUtil.add_maxpool_layer(outputs)

            tscope = "carriage_" + str(it)
            outputs = ModelUtil.add_building_block_carriage(
                batch_num, self.hps.deepk[it], carriage_block_num[it], outputs,
                kernalWidth, is_training_ph,
                scope=tscope, layername=layername, activateFunc=activateFunc,
                stride=stride)

        return outputs

    def create_resnet(self, inputx, is_training_ph, activateFunc=tf.nn.relu):

        reshead = self.create_deep_res_head(
            inputx, is_training_ph, activateFunc=activateFunc)

        stride = [1,1,1,1]
        outputs = self.create_deep_res_body(
            self.hps.batch_nums, self.hps.carriage_block_num, reshead,
            is_training_ph, activateFunc=activateFunc,stride=stride)

        ###########################
        print("final outputs shape:%s" % str(outputs.get_shape().as_list()))
        # print "before fractional outputs shape:", outputs.get_shape().as_list()
        # outputs = ModelUtil.add_fractional_maxpool(outputs,ratio=[1,1.4,1.4,1])
        # print "after fractional outputs shape:",outputs.get_shape().as_list()
        ###########################

        return outputs

    def create_graph(self, MAX_CHARACTER_LENGTH):
        """
        :param MAX_CHARACTER_LENGTH: 最大字符串长度
        """
        # create graph start
        self.is_training_ph = tf.placeholder(tf.bool)


        resnet = self.create_resnet(self._images, self.is_training_ph, activateFunc=tf.nn.relu)
        fcl1_inputs, fcl1_in_features = ModelUtil.conv2fc(resnet)
        # set outputs features
        outputs_features = (self.hps.num_classes + 1) * MAX_CHARACTER_LENGTH
        self.outputs = ModelUtil.add_fc_layer(fcl1_inputs, fcl1_in_features, outputs_features)

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.outputs, labels=self.labes))

        if self.mode == "train":
            self.learning_rate = tf.placeholder(tf.float32)
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss=self.loss)

        
        
