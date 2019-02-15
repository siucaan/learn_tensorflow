import tensorflow as tf

# Create some variables.
v1 = tf.get_variable(name="v1", shape=[3], initializer=tf.zeros_initializer)
v2 = tf.get_variable(name="v2", shape=[5], initializer=tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# 不加参数时，默认保存所有变量
saver = tf.train.Saver()
# 列表或者字典的形式传入想要保存的变量
# saver = tf.train.Saver([v1])
# 可以指定想要保存的模型个数
# saver = tf.train.Saver(max_to_keep=4,)
# 指定训练过程的什么时候保存，如2小时保存一次
# saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

# launch the model, initialize the variables, do some work, and save the variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  # 打印输出
  print(v1.eval())
  print(v2.eval())
  # Save the variables to disk.
  saver.save(sess, "./save_restore_model/my_test")  # 保存到当前目录下的save_restore_model文件夹，文件名为‘my_test.XXX’
  # save model every 1000 iterations
  # saver.save(sess, "test_save_restore_model", global_step=1000, write_meta_graph=False)

