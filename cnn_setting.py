# -*- coding:utf-8 -*-

import tensorflow as tf
import sys
if sys.argv[0] == "cnn_train.py":
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])

def weight_variable(shape):
	init = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(init)

def bias_variable(shape):
	init = tf.constant(0.1,shape=shape)
	return tf.Variable(init)
	

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

#１件分のデータを　２８x２８x１に変換
x_image = tf.reshape(x, [-1,28,28,1])

#フィルターは、５x５x１x３２（H=5, K=1, M=32）
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#畳み込みそう、プーリング層を定義。　relu関数で活性化
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#フィルターを５x５x３２x６４定義。
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#畳み込みそう、プーリング層を定義。　relu関数で活性化
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#全結合の第一層のweightとバイアスを定義
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

#プーリング層からの出力結果を７x７x６４のサイズに変更
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#全結合の第一層目を出力
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#ドロップアウトの定義
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#全結合第２層を定義
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

#ソフトマックスでクラス分類、以下、全結合のモデル
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

