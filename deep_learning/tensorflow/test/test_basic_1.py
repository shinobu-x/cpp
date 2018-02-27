import tensorflow as tf

# y = x^2 + a
x1 = tf.constant(3.)
a1 = tf.constant(2.)
r1 = tf.square(x1)
r1 = tf.add(r1, a1)
with tf.Session() as s:
  print(s.run(r1))

x2 = tf.placeholder(tf.float32)
a2 = tf.placeholder(tf.float32)
f1 = tf.add(tf.square(x2), a2)
with tf.Session() as s:
  r = s.run([f1], feed_dict={x2: [2.], a2: [3.]})
  print(r)
