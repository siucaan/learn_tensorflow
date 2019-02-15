import tensorflow as tf


def return_variable():
    # 形状为2*3的正态分布，均值为0，标准差为2; seed设定后每次随机生成的值相同
    al = []
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=2, seed=1))

    # 通过tf.global_variables_initializer函数全部初始化
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        al.append(w1.eval())

    return al


if __name__ == '__main__':
    print(return_variable())