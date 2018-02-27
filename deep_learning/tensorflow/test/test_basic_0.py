from __future__ import print_function

import tensorflow as tf

x = tf.constant(2)
y = tf.constant(3)
with tf.Session() as s:
  print(s.run(x + y))
  print(s.run(x * y))

x = tf.placeholder(tf.int16)
y = tf.placeholder(tf.int16)
add = tf.add(x, y)
mul = tf.add(x, y)
with tf.Session() as s:
  print(s.run(add, feed_dict = {x: 2, y: 3}))
  print(s.run(mul, feed_dict = {x: 2, y: 3}))

m1 = tf.constant([[3., 3.]])
m2 = tf.constant([[2.], [2.]])
p = tf.matmul(m1, m2)
with tf.Session() as s:
  print(s.run(p))
