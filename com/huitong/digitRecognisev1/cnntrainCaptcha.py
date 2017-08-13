#!/usr/bin/env python
#encoding=utf-8

"""
@author ZHAOPENGCHENG on 2017/8/9.
"""
from com.huitong.digitRecognisev1.genCaptcha import gen_captcha_text_and_image
from com.huitong.digitRecognisev1.genCaptcha import number


import numpy as np
import tensorflow as tf

text, image = gen_captcha_text_and_image() #先生成验证码和文字测试模块是否完全
print("chaptcha picture channel:", image.shape)  # (60, 160, 3)
# 图像大小
IMAGE_HEIGHT = image.shape[0]
IMAGE_WIDTH = image.shape[1]
IMAGE_DEPTH_ORIGN = image.shape[2]
IMAGE_DEPTH_DES = 1

MAX_CAPTCHA = len(text)

# conv layer feature depth
convLayerOutchannals = [32, 64, 64]
print("max length of chaptcha character", MAX_CAPTCHA)   # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐

saveVariableDirectory = "/home/allen/work/data/digitalRecognise/v1"
BESTACC = 0

# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
	if len(img.shape) > 2:
		gray = np.mean(img, -1)
		# 上面的转法较快，正规转法如下
		# r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
		# gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
		return gray
	else:
		return img

"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
np.pad(image【,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
"""

# 文本转向量
char_set = number + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(char_set) #字符集中字符数量
def text2vec(text):
	"""
	将验证码字符串转换成one hot矢量
	:param text:
	:return:
	"""
	text_len = len(text)
	if text_len > MAX_CAPTCHA:
		raise ValueError('max length of captcha character is 4')

	vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
	def char2pos(c):
		if c =='_':
			k = 10
			return k
		if c >= '0' and c <= '9':
			k = ord(c) - ord('0')
		else:
			raise Exception('character not in 0-9')
		return k
	for i, c in enumerate(text):
		idx = i * CHAR_SET_LEN + char2pos(c)
		vector[idx] = 1
	return vector
# 向量转回文本
def vec2text(vec):
	"""
	将 one hot 矢量转换成字符串
	:param vec:
	:return:
	"""
	char_pos = vec.nonzero()[0]
	text=[]
	for i, c in enumerate(char_pos):
		char_idx = c % CHAR_SET_LEN
		if char_idx < 10:
			char_code = char_idx + ord('0')
		elif char_idx == 10:
			char_code = ord('_')
		else:
			raise ValueError('error')
		text.append(chr(char_code))
	return "".join(text)

"""
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有
vec = text2vec("F5Sd")
text = vec2text(vec)
print(text)  # F5Sd
vec = text2vec("SFd5")
text = vec2text(vec)
print(text)  # SFd5
"""

# 生成一个训练batch
def get_next_batch(batch_size=128):
	batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH_DES])
	batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])

	# 有时生成图像大小不是(60, 160, 3)
	def wrap_gen_captcha_text_and_image():
		''' 获取一张图，判断其是否符合（60，160，3）的规格'''
		while True:
			text, image = gen_captcha_text_and_image()
			if image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH_ORIGN):#此部分应该与开头部分图片宽高吻合
				return text, image

	for i in range(batch_size):
		text, image = wrap_gen_captcha_text_and_image()
		image = convert2gray(image)

		# 将图片数组一维化 同时将文本也对应在两个二维组的同一行
		batch_x[i,:] = image.flatten() / 255 # (image.flatten()-128)/128  mean为0
		batch_y[i,:] = text2vec(text)
	# 返回该训练批次
	return batch_x, batch_y

####################################################################
# 申请占位符 按照图片
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH_DES])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32) # dropout

# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
	# 将占位符 转换为 按照图片给的新样式
	x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH_DES])

	# 3 conv layer
	w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, IMAGE_DEPTH_DES, convLayerOutchannals[0]])) # 从正太分布输出随机值
	b_c1 = tf.Variable(b_alpha*tf.random_normal([convLayerOutchannals[0]]))
	conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
	conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv1 = tf.nn.dropout(conv1, keep_prob)

	w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, convLayerOutchannals[0], convLayerOutchannals[1]]))
	b_c2 = tf.Variable(b_alpha*tf.random_normal([convLayerOutchannals[1]]))
	conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
	conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv2 = tf.nn.dropout(conv2, keep_prob)

	w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, convLayerOutchannals[1], convLayerOutchannals[2]]))
	b_c3 = tf.Variable(b_alpha*tf.random_normal([convLayerOutchannals[2]]))
	conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
	conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv3 = tf.nn.dropout(conv3, keep_prob)

	# Fully connected layer
	w_d = tf.Variable(w_alpha*tf.random_normal([8*20*convLayerOutchannals[2], 1024]))
	b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
	dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
	dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
	dense = tf.nn.dropout(dense, keep_prob)

	w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))
	b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
	out = tf.add(tf.matmul(dense, w_out), b_out)
	#out = tf.nn.softmax(out)
	return out

# 训练
def train_crack_captcha_cnn():
	output = crack_captcha_cnn()
	#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
        # 最后一层用来分类的softmax和sigmoid有什么不同？
	# optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

	predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
	max_idx_p = tf.argmax(predict, 2)
	max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
	correct_pred = tf.equal(max_idx_p, max_idx_l)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		step = 1
		while True:
			batch_x, batch_y = get_next_batch(64)
			_, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})

			if(step % 10 == 0):
				print ('step = %d, loss = %.4f' % (step, loss_))

			# 每100 step计算一次准确率
			if step % 400 == 0:
				batch_x_test, batch_y_test = get_next_batch(100)
				acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
				print('step = %d, acc = %.4f' % (step, acc))
				# 如果准确率大于50%,保存模型,完成训练
				if acc > BESTACC:
					BESTACC = acc
					# saver.save(sess, "crack_capcha.model", global_step=step)
					saver.save(sess, save_path=saveVariableDirectory)
			step += 1

def startTrain():
	train_crack_captcha_cnn()



if __name__ == "__main__":
	startTrain()
	# fun1()
