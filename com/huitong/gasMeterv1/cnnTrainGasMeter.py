#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/11.
"""
from com.huitong.gasMeterv1.genDigitPic import GenDigitPicture

import tensorflow as tf
import os

captchaCharacterLength = 4
captchaBoxWidth = 128
captchaBoxHeight = 64

gen = GenDigitPicture(captchaCharacterLength,captchaBoxWidth,captchaBoxHeight)

text, image = gen.get_text_and_image()  # 先生成验证码和文字测试模块是否完全
print("chaptcha picture channel:", image.shape)  # (60, 160, 3)


# 图像大小
IMAGE_HEIGHT = gen.ImageHeight
IMAGE_WIDTH = gen.ImageWidth
IMAGE_DEPTH = gen.ImageDepth

MAX_CAPTCHA_CHARACTER_LENGTH = gen.Max_Character_Length

# conv layer feature depth
convLayerOutchannals = [32, 64, 64]
print("max length of chaptcha character", MAX_CAPTCHA_CHARACTER_LENGTH)  # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐

saveVariableDirectory = ["data","digitRecognise","v1"]
saveFileTempName = "temp.ckpy"

BESTACC = 0


"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
np.pad(image【,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
"""

# 文本转向量
CHAR_SET_LEN = len(gen.CharSet) + 1   # 字符集中字符数量




####################################################################
# 申请占位符 按照图片
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA_CHARACTER_LENGTH * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout


# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    # 将占位符 转换为 按照图片给的新样式
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, IMAGE_DEPTH, convLayerOutchannals[0]]))  # 从正太分布输出随机值
    b_c1 = tf.Variable(b_alpha * tf.random_normal([convLayerOutchannals[0]]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, convLayerOutchannals[0], convLayerOutchannals[1]]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([convLayerOutchannals[1]]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, convLayerOutchannals[1], convLayerOutchannals[2]]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([convLayerOutchannals[2]]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 16 * convLayerOutchannals[2], 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA_CHARACTER_LENGTH * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA_CHARACTER_LENGTH * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out


# 训练
def train_crack_captcha_cnn():
    global BESTACC, saveFileTempName
    saveFileTempName = processSaveVariavlepath(saveVariableDirectory, saveFileTempName)

    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA_CHARACTER_LENGTH, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA_CHARACTER_LENGTH, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 1
        while True:
            batch_x, batch_y = gen.get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})

            if step % 50 == 0:
                print('step = %d, loss = %.4f' % (step, loss_))

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = gen.get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print('step = %d, acc = %.4f' % (step, acc))
                # 如果准确率大于50%,保存模型,完成训练
                if acc > BESTACC:
                    BESTACC = acc
                    saver.save(sess, save_path=saveFileTempName)
            step += 1

def processSaveVariavlepath(dirnames, filename):
    path = os.path.dirname(__file__)
    for each in dirnames:
        path = os.path.join(path, each)
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, filename)
    return filename


def startTrain():
    train_crack_captcha_cnn()


if __name__ == "__main__":
    startTrain()