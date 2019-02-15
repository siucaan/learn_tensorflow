import tensorflow as tf

# 形状为2*3的正态分布，均值为0，标准差为2; seed设定后每次随机生成的值相同
weights1 = tf.Variable(tf.random_normal([2, 3], stddev = 2, seed = 1))
# 形状为2*3的正态分布，均值为0，标准差为2; seed设定后每次随机生成的值不相同
weights2 = tf.Variable(tf.random_normal([2, 3], mean = 1, stddev = 2))
# 使用常数来设置偏置项（bias）初始值； 生成长度为3，值为0
biases = tf.Variable(tf.zeros([3]))
# 通过其他变量设置初始值
w2 = tf.Variable(weights1.initialized_value())

# 通过tf.global_variables_initializer函数全部初始化
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print("weights1=", sess.run(weights1))
    print("weights2=", sess.run(weights2))
    print("biases=", sess.run(biases))
    print("w2=", sess.run(w2))
