#coding=utf-8
"""
该文件就是tensorflowproject1中tools package中的OCRUtilv3s2文件
MyLog is to log result
learning_rate_down is to down learning rate

all character
test data all 3755 class, character number is:   533675
train data all 3755 class, character number is: 2144749

test data 100 class, character number is:  14202
train data 100 class, character number is: 56987

"""
import logging,os,pickle
import shutil
import tensorflow as tf
from tensorflow.contrib.layers.python.layers.layers import batch_norm
import numpy as np
from imgaug import augmenters as iaa
import random

class MyLog(object):
    logfile = ""
    logger = None

    def __init__(self,logfile):
        self.logfile = logfile
        filehandler = logging.FileHandler(filename=logfile,encoding='utf-8')
        fmter = logging.Formatter(fmt="%(asctime)s %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
        filehandler.setFormatter(fmter)
        loger = logging.getLogger(__name__)
        loger.addHandler(filehandler)
        loger.setLevel(logging.DEBUG)
        self.logger = loger

    def log_message(self,msg):
        self.logger.debug(msg)

    def showAndLogMsg(self,msg):
        self.log_message(msg)
        print msg


def init_peizhi(peizhifilename,peizhidict):
    if not os.path.exists(peizhifilename):
        with open(peizhifilename, mode='w') as wfobj:
            pickle.dump(peizhidict,wfobj)

def update_peizhi(peizhi_filename, key, value):
    """
    更新配置文件中的配置项
    """
    with open(peizhi_filename) as rfobj:
        peizhi = pickle.load(rfobj)
    peizhi[key] = value
    with open(peizhi_filename, mode='w') as wfobj:
        pickle.dump(peizhi,wfobj)

def get_peizhi_val(peizhifilename,key):
    with open(peizhifilename) as rfobj:
        peizhi = pickle.load(rfobj)
    return peizhi[key]

def get_filename(basedir,dirindex,fileindex):
    desfilename = os.path.join(basedir,"{:0>2}".format(str(dirindex)))
    tfilename = "{:0>5}".format(str(fileindex)) + "-c.gnt"
    desfilename = os.path.join(desfilename,tfilename)
    return desfilename

def get_accurate(prediction,labels):
    equalFlags = np.equal(np.argmax(prediction,axis=1),np.argmax(labels,axis=1))
    return np.mean(equalFlags)


def get_test_right_num(prediction,labels):
    return np.sum(np.equal(np.argmax(prediction,axis=1),np.argmax(labels,axis=1)))

def add_fc_layer(
        inputs, inFeatures, outFeatures, layerName="layer", activateFunc=None):
    with tf.name_scope(layerName):
        Weights = tf.Variable(tf.truncated_normal([inFeatures, outFeatures], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, tf.float32, [outFeatures]))

        y = tf.matmul(inputs,Weights) + biases
        if activateFunc is None:
            outputs = y
        else:
            outputs = activateFunc(y)

        return outputs



def add_building_block_carriage(batch_num,deepk, carriage_nums, inputs, kernalWidth,
                                is_training_ph, scope=None, layername="layer",
                                activateFunc=tf.nn.relu, stride=[1, 1, 1, 1]):
    """
    conv layer number: 2*carriage_nums
    将一组building_bolck组合在一起,形成一串
    the depth of outputs is same as inDepth
    :param carriage_nums:要添加的building_bolck数目
    """
    if scope is None:
        raise ValueError('scope should be a string')
    if carriage_nums <1:
        raise ValueError('nums should not be less than 1')

    tscope = scope + "blockincre"
    outputs = building_block_incre(batch_num,deepk,inputs,kernalWidth,is_training_ph,scope=tscope,
                                   layername=layername,activateFunc=activateFunc,stride=stride)

    for it in range(1, carriage_nums):
        tscope = scope + "block" + str(it)
        outputs = building_block_same(outputs,kernalWidth,is_training_ph,scope=tscope,layername=layername,
                                      activateFunc=activateFunc,stride=stride)

    return outputs

def building_block_incre(batch_num,deepk,inputs,kernalWidth,
                         is_training_ph,scope=None, layername="layer",
                         activateFunc=tf.nn.relu, stride=[1, 1, 1, 1]):
    """
    conv layer number: 3
    首先是一层max pooling层,,两层卷积层,和project short cut connection 层组成
    在增加维度时short cut connection使用projection
    the depth of outputs is 2*inDepth
    """
    if scope is None:
        raise ValueError('scope should be a string')
    if deepk < 1:
        raise ValueError('deepk should not be less than 1')

    inshape = inputs.get_shape().as_list()
    inDepth = inshape[3]
    inshape[0] = batch_num

    padshape = inshape[:]
    depth = int(deepk*inDepth)
    padshape[-1] = depth - inshape[-1]

    zero_pad = np.zeros(padshape,np.float32)
    proj = tf.concat(values=[inputs,zero_pad],axis=3)

    kw = 1

    tscope = scope + "layer1"
    y = add_BN_conv_layer(inputs,kw,inDepth,inDepth,
                           is_training_ph,tscope,
                           activateFunc=activateFunc,stride=stride)

    tscope = scope + "layer2"
    y = add_BN_conv_layer(y,kernalWidth,inDepth,inDepth,
                           is_training_ph,tscope,
                           activateFunc=None,stride=stride)

    tscope = scope + "layer3"
    y = add_BN_conv_layer(y, kw, inDepth, depth,
                          is_training_ph, tscope,
                          activateFunc=activateFunc,stride=stride)

    hx = tf.add(y, proj)
    outputs = activateFunc(hx)
    return outputs

def building_block_same(inputs,kernalWidth,
                        is_training_ph, scope=None, layername="layer",
                        activateFunc=tf.nn.relu, stride=[1, 1, 1, 1]):
    """
    conv layer number:3
    """
    if scope is None:
        raise ValueError('scope should be a string')
    inDepth = inputs.get_shape().as_list()[3]
    depth = inDepth/2

    kw = 1

    tscope = scope + "layer1"
    y = add_BN_conv_layer(inputs,kw,inDepth,depth,
                           is_training_ph,tscope,
                           activateFunc=activateFunc,stride=stride)

    tscope = scope + "layer2"
    y=add_BN_conv_layer(y, kernalWidth, depth, depth,
                        is_training_ph, tscope,
                        activateFunc=activateFunc,stride=stride)

    tscope = scope + "layer3"
    y = add_BN_conv_layer(y,kw,depth,inDepth,
                           is_training_ph,tscope,
                           activateFunc=None,stride=stride)

    hx = tf.add(y, inputs)
    outputs = activateFunc(hx)
    return outputs

def building_block_desc(inputs,is_training_ph, scope=None, layername="layer",
                        activateFunc=None, stride=[1, 1, 1, 1],
                        descrate=0.6):
    """
    conv layer:1
    压缩率：descrate
    """

    inDepth = inputs.get_shape().as_list()[3]
    outDepth = int(inDepth*descrate)
    tscope = scope + "layer1"

    kw=1
    outputs = add_BN_conv_layer(inputs,kw,inDepth,outDepth,is_training_ph,
                                tscope,layername=layername,activateFunc=activateFunc,
                                stride=stride)
    return outputs


def add_BN_conv_layer(inputs, kernalWidth, inDepth, outDepth,
                      is_training_ph, scope , layername="layer",
                      activateFunc=tf.nn.relu, stride=[1, 1, 1, 1]):

    # inDepth = inputs.get_shape().as_list()[3]

    with tf.name_scope(layername):
        n = kernalWidth * kernalWidth * outDepth
        Weights = tf.Variable(tf.truncated_normal([kernalWidth, kernalWidth, inDepth, outDepth], stddev=np.sqrt(2.0/n)))
        biases = tf.Variable(tf.constant(0.1, tf.float32, [outDepth]))

        y1 = tf.nn.conv2d(inputs, Weights, stride, padding='SAME') + biases

        outputs = tf.cond(is_training_ph,
                           lambda: batch_norm(y1,decay=0.94, is_training=True,
                                              center=False, scale=True,
                                              activation_fn=activateFunc,
                                              updates_collections=None, scope=scope),
                           lambda: batch_norm(y1,decay=0.94, is_training=False,
                                              center=False, scale=True,
                                              activation_fn=activateFunc,
                                              updates_collections=None, scope=scope,
                                              reuse=True))

        return outputs

def add_overlap_maxpool(inputs,kernal=3,step=2,layername="poollayer"):
    with tf.name_scope(layername):
        kernals = [1,kernal,kernal,1]
        strides = [1,step,step,1]
        return tf.nn.max_pool(inputs,kernals,strides,padding='SAME')

def add_overlap_avgpool(inputs,kernal=3,step=2,layername="poollayer"):
    with tf.name_scope(layername):
        kernals = [1,kernal,kernal,1]
        strides = [1,step,step,1]
        return tf.nn.max_pool(inputs,kernals,strides,padding='SAME')

def add_maxpool_layer(inputs,step=2,layername="poolLayer"):
    with tf.name_scope(layername):
        kernal = [1, step, step, 1]
        return tf.nn.max_pool(inputs,kernal,strides=kernal,padding='SAME')


def add_averagepool_layer(inputs,step=2,layername="poolLayer"):
    with tf.name_scope(layername):
        kernal = [1, step, step, 1]
        return tf.nn.avg_pool(inputs,kernal,strides=kernal,padding='SAME')

def add_fractional_maxpool(inputs, ratio = None,layername = "fractional_max_pool"):
    with tf.name_scope(layername):
        if ratio is None:
            ratios = [1, 1.4, 1.4, 1]
        else:
            ratios = ratio
        outputs,row_pool,col_pool = tf.nn.fractional_max_pool(inputs, ratios)
        return outputs

def add_fractional_avgpool(inputs, ratio = None, layername = 'fractional_avg_pool'):
    with tf.name_scope(layername):
        if ratio is None:
            ratios = [1,1.4,1.4,1]
        else:
            ratios = ratio

        outputs, row_pool, col_pool = tf.nn.fractional_avg_pool(inputs,ratios)
        return outputs




def conv2fc(inputs):
    conv_out_shape = inputs.get_shape().as_list()
    fcl_in_features = conv_out_shape[1] * conv_out_shape[2] * conv_out_shape[3]
    fcl_inputs = tf.reshape(inputs, [-1, fcl_in_features])
    return fcl_inputs,fcl_in_features

def leaky_relu(X,alpha=0.2,name="LeakyRelu"):
    with tf.name_scope(name):
        X = tf.maximum(X,alpha*X)
    return X

def randomized_relu(X, mode = 'train', name="RandomizedRelu"):
    alpha = random.uniform(1.0/8, 1.0/3)
    with tf.name_scope(name):
        if mode == "train":
            X = tf.maximum(X, alpha*X)
        else:
            X = tf.maxmum(X, 2.0/11*X)

    return X


def down_learning_rate(test_acc, lr):
    if lr <= 5e-5:
        return 5e-5

    if test_acc >= 0.8 :
        lr /= 5.0

    return lr

def empty_dir(dirname):
    if not os.path.exists(dirname):
        raise ValueError('%s dir not exist'%dirname)

    shutil.rmtree(dirname)
    os.mkdir(dirname)
    return True

def move_variable_from_src2des(srcdirname,desdirname):
    empty_dir(desdirname)
    for each in os.listdir(srcdirname):
        orifilename = os.path.join(srcdirname,each)
        desfilename = os.path.join(desdirname,each)
        shutil.copy(orifilename,desfilename)


def test():
    # loger = MyLog("/home/allen/work/temp/test.txt")
    # loger.log_message("nice to meet you")
    # num = 0
    #
    # lr = 2e-3
    # acc = 0.85
    # while True:
    #     num += 1
    #     lr = down_learning_rate(acc,lr)
    #     print "%d %f"%(num,lr)
    #     if lr < 1e-4:
    #         acc = 0.95
    #
    #     if lr < 1e-5:
    #         acc = 0.96
    dirname = "/home/allen/work/variableSave/0temp"
    empty_dir(dirname)
    print "done"





if __name__ == "__main__":
    test()


