#coding=utf-8
"""

"""
import logging,os,pickle
import shutil
import tensorflow as tf
from tensorflow.contrib.layers.python.layers.layers import batch_norm
import numpy as np

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
        print(msg)


def init_peizhi(peizhifilename,peizhidict):
    if not os.path.exists(peizhifilename):
        with open(peizhifilename, mode='wb') as wfobj:
            pickle.dump(peizhidict,wfobj)

def update_peizhi(peizhi_filename, key, value):
    """
    更新配置文件中的配置项
    """
    with open(peizhi_filename,mode='rb') as rfobj:
        peizhi = pickle.load(rfobj)
    peizhi[key] = value
    with open(peizhi_filename, mode='wb') as wfobj:
        pickle.dump(peizhi,wfobj)

def get_peizhi_val(peizhifilename,key):
    with open(peizhifilename,mode='rb') as rfobj:
        peizhi = pickle.load(rfobj)
    return peizhi[key]

def get_filename(basedir,dirindex,fileindex):
    desfilename = os.path.join(basedir,"{:0>2}".format(str(dirindex)))
    tfilename = "{:0>5}".format(str(fileindex)) + "-c.gnt"
    desfilename = os.path.join(desfilename,tfilename)
    return desfilename

def get_str_accurate(outputs,labels, MAX_CHARACTER_LENGTH, CHAR_SET_LEN):
    predict = np.reshape(outputs, [-1, MAX_CHARACTER_LENGTH, CHAR_SET_LEN])
    max_idx_p = np.argmax(predict, 2)
    max_idx_l = np.argmax(np.reshape(labels, [-1, MAX_CHARACTER_LENGTH, CHAR_SET_LEN]), 2)
    correct_pred = np.equal(max_idx_p, max_idx_l)
    accuracy = np.mean(correct_pred)
    return accuracy

def get_test_right_num(prediction,labels):
    return np.sum(np.equal(np.argmax(prediction,axis=1),np.argmax(labels,axis=1)))

def add_fc_layer(
        inputs, inFeatures, outFeatures, layerName="layer", activateFunc=None):
    with tf.name_scope(layerName):
        Weights = tf.Variable(tf.truncated_normal([inFeatures, outFeatures], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, tf.float32, [outFeatures]))

        # y = tf.matmul(inputs,Weights) + biases
        y = tf.nn.xw_plus_b(inputs, Weights, biases)
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
    depth = int(inDepth/2)

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

def add_maxpool_layer(inputs,step=2,layername="poolLayer"):
    with tf.name_scope(layername):
        kernal = [1, step, step, 1]
        return tf.nn.max_pool(inputs,kernal,strides=kernal,padding='SAME')


def conv2fc(inputs):
    conv_out_shape = inputs.get_shape().as_list()
    fcl_in_features = conv_out_shape[1] * conv_out_shape[2] * conv_out_shape[3]
    fcl_inputs = tf.reshape(inputs, [-1, fcl_in_features])
    return fcl_inputs,fcl_in_features



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

def setTrainstep():
    filename = "peizhi.xml"
    k = "train_step"
    v = 1001

    val = get_peizhi_val(filename,k)
    print(val)


def test():
    setTrainstep()





if __name__ == "__main__":
    test()


