import tensorflow as tf 

hello_constant = tf.constant('hello world')
pl = tf.placeholder(tf.int32)

with tf.Session() as sess:
	output = sess.run(hello_constant)
	output = sess.run(pl, feed_dict={pl: 1})
	print(output)
	output = sess.run(pl, feed_dict={pl: 2})
	print(output)