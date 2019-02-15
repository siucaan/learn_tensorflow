import numpy as np
import tensorflow as tf


def save_tfrecords(data, label, desfile):
    with tf.python_io.TFRecordWriter(desfile) as writer:
        for i in range(len(data)):
            features = tf.train.Features(
                feature = {
                    "data":tf.train.Feature(bytes_list=tf.train.BytesList(value=[data[i].astype(np.float32).tostring()])),
                    "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[label[i]]))
                }
            )
            example = tf.train.Example(features = features)
            serialized = example.SerializeToString()
            writer.write(serialized)


def padding(data, maxlen=10):
    for i in range(len(data)):
        data[i] = np.hstack([data[i], np.zeros((maxlen-len(data[i])))])


def _parse_function(example_proto):
  features = {"data": tf.FixedLenFeature((), tf.string),
              "label": tf.FixedLenFeature((), tf.int64)}
  parsed_features = tf.parse_single_example(example_proto, features)
  data = tf.decode_raw(parsed_features['data'], tf.float32)

  return data, parsed_features["label"]


def load_tfrecords(srcfile):
    sess = tf.Session()
    dataset = tf.data.TFRecordDataset(srcfile) # load tfrecord file
    dataset = dataset.map(_parse_function) # parse data into tensor
    dataset = dataset.repeat(2) # repeat for 2 epoches
    dataset = dataset.batch(4) # set batch_size = 5

    iterator = dataset.make_one_shot_iterator()
    next_data = iterator.get_next()

    while True:
        try:
            data, label = sess.run(next_data)
            print(data)
            print(label)
        except tf.errors.OutOfRangeError:
            break


if __name__ =='__main__':

    lens = np.random.randint(low=3, high=10, size=(10,))
    data = [np.arange(l) for l in lens]
    padding(data)
    print(data)
    label = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    save_tfrecords(data, label, "./data.tfrecords")
    load_tfrecords(srcfile="./data.tfrecords")
    pass
