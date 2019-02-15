import tensorflow as tf

#从ckpt文件中获取variable变量的名字
def get_trainable_variables_name_from_ckpt(meta_graph_path,ckpt_path):
    #定义一个新的graph
    graph = tf.Graph()
    #将其设置为默认图:
    with graph.as_default():
        with tf.Session() as session:
            #加载计算图
            saver = tf.train.import_meta_graph(meta_graph_path)
            #加载模型到session中关联的graph中，即将模型文件中的计算图加载到这里的graph中
            saver.restore(session, ckpt_path)
            v_names = []
            #获取session所关联的图中可被训练的variable
            #使用tf.trainable_variables()获取variable时，只有在该函数前面定义的variable才会被获取到
            #在其后面定义不会被获取到，
            for v in tf.trainable_variables():
                v_names.append(v)
            return v_names
#利用pywrap_tensorflow获取ckpt文件中的所有变量，得到的是variable名字与shape的一个map

from tensorflow.python import pywrap_tensorflow
def get_all_variables_name_from_ckpt(ckpt_path):
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    all_var = reader.get_variable_to_shape_map()
    #reader.get_variable_to_dtype_map()
    return all_var


#从cpkt文件中拷贝模型的参数到自定义的变量中
def copy_var_from_ckpt(session, dst_var_name, dst_var,ckpt_path, meta_graph_path):
    #定义一个新的graph
    graph = tf.Graph()
    #将其设置为默认图:
    with graph.as_default():
        with tf.Session() as sess:
            #加载计算图
            saver = tf.train.import_meta_graph(meta_graph_path)
            #加载模型到session中关联的graph中，即将模型文件中的计算图加载到这里的graph中
            saver.restore(sess,ckpt_path)
            v_names = []
            #获取session所关联的图中可被训练的variable
            #使用tf.trainable_variables()获取variable时，只有在该函数前面定义的variable才会被获取到
            #在其后面定义不会被获取到，
            for v in tf.trainable_variables():
                v_names.append(v)
            if dst_var_name in v_names:
                #获取tensor
                tensor = graph.get_tensor_by_name(dst_var_name)
                #获取tensor的值，即网络中权值
                weight = sess.run(tensor)
                #拷贝权值,注意，需要使用dst_var所在的session
                #使用assign操作来拷贝dst_var是一个variable，weight是一个array
                session.run(dst_var.assign(weight))


if __name__ == '__main__':
    meta_graph_path = "./save_restore_model/my_test.meta"
    ckpt_path = tf.train.latest_checkpoint('./save_restore_model')
    # v_names = get_trainable_variables_name_from_ckpt(meta_graph_path, ckpt_path)
    # print(v_names)

    all_var = get_all_variables_name_from_ckpt(ckpt_path)
    print(all_var)

