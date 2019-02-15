import tensorflow as tf
import numpy as np

a = tf.constant([1, 2, 3, 4, 5, 6, 7])
b = tf.constant(-1.0, shape=[2, 3])
c = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
d = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
e = tf.constant(np.arange(1, 13, dtype=np.int32), shape=[2, 2, 3])

with tf.Session() as sess:
    print("a =", sess.run(a))
    print("b =", sess.run(b))
    print("c =", sess.run(c))
    print("d =", sess.run(d))
    print("e =", sess.run(e))