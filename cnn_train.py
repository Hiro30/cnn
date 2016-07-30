# -*- coding:utf-8 -*-
import tensorflow as tf
#変数定義。
import cnn_setting
import os.path

x_entropy = tf.reduce_mean(-tf.reduce_sum(cnn_setting.y_ * tf.log(cnn_setting.y), reduction_indices=[1]))

optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(x_entropy)

init = tf.initialize_all_variables()

saver = tf.train.Saver()

sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(cnn_setting.y,1), tf.argmax(cnn_setting.y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if os.path.isfile("model.ckpt") == True:
	saver.restore(sess, "model.ckpt")

for i in range(1, 1001):
	batch_xs, batch_ys = cnn_setting.mnist.train.next_batch(50)
	sess.run(train_step, feed_dict={cnn_setting.x: batch_xs, cnn_setting.y_: batch_ys, cnn_setting.keep_prob: 0.5})
	if i % 200 == 0:
		train_accuracy = sess.run(accuracy, feed_dict={cnn_setting.x: batch_xs, cnn_setting.y_: batch_ys, cnn_setting.keep_prob: 1.0})
		print('  step, accurary = %6d: %6.3f' % (i, train_accuracy))

print(sess.run(accuracy, feed_dict={cnn_setting.x: cnn_setting.mnist.test.images,cnn_setting.y_: cnn_setting.mnist.test.labels,cnn_setting.keep_prob: 1.0}))
saver.save(sess, "model.ckpt")
sess.close()
