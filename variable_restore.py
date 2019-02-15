import tensorflow as tf
tf.reset_default_graph()
# # Create some variables.
# v1 = tf.get_variable("v1", [3], initializer = tf.zeros_initializer)
# v2 = tf.get_variable("v2", [5], initializer = tf.zeros_initializer)
#
# # 恢复所有变量
# saver = tf.train.Saver()
# # 恢复变量v2
# # saver = tf.train.Saver({"v2": v2})
#
# # Use the saver object normally after that.
# with tf.Session() as sess:
#   # Initialize v1 since the saver will not.
#   v1.initializer.run()
#   saver.restore(sess, "./save_restore_model/my_test")
#
#   print("v1 : %s" % v1.eval())
#   print("v2 : %s" % v2.eval())
#
# with tf.Session() as sess:
#     new_saver = tf.train.import_meta_graph('./save_restore_model/my_test.meta')
#     new_saver.restore(sess, "./save_restore_model/my_test")
#     print(sess.run('v1:0'))

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./save_restore_model/my_test.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./save_restore_model'))
    print(sess.run('v1:0'))

