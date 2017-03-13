import tensorflow as tf

a = tf.constant([[1, 9, 4], [1, 0, 3]], dtype=tf.float32)
b = tf.constant([[0, 0, 2], [2, 0, 0]], dtype=tf.float32)

c = tf.reduce_sum(a * b, 1) / tf.sqrt(tf.reduce_sum(tf.square(a), 1)) / tf.sqrt(tf.reduce_sum(tf.square(b), 1))


d = tf.sqrt( tf.reduce_sum(tf.square(tf.sub(a, b)), reduction_indices=1))
# a2 = tf.sqrt(tf.reduce_sum(tf.square(a), 1))
# b2 = tf.sqrt(tf.reduce_sum(tf.square(b), 1))
#
# a3 = tf.div(c, a2)
# b3 = tf.div(a3, b2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(c))
print(sess.run(d))
# print(sess.run(a2))
# print(sess.run(b2))
# print(sess.run(a3))
# print(sess.run(b3))
