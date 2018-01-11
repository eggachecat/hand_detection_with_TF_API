import io
import json
import os
import skimage
import skimage.io
import skimage.transform
import scipy.misc
from skimage.viewer import ImageViewer
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util

DATA_PATH = ""
TRAIN_PATH = "./training/train/"
EVAL_PATH = "./training/eval/"
TRAIN_RATIO = 0.8

base_path_list = []
prefix_list = []


def generate_paths_0():
    global base_path_list, prefix_list
    """for synth data"""
    for i in range(2):
        base_path = os.path.join(DATA_PATH, "DeepQ-Synth-Hand-0{}/data".format(i + 1))
        for j in range(5):
            base_path_list.append(os.path.join(base_path, "s00{}".format(i * 5 + j)))
            prefix_list.append("{}".format(i * 5 + j))

    """for real data"""

    base_path_list += [os.path.join(DATA_PATH, "DeepQ-Vivepaper/data/air"),
                       os.path.join(DATA_PATH, "DeepQ-Vivepaper/data/book")]
    prefix_list += [10, 11]


def generate_paths_1():
    """
    for Ren
    :return:
    """
    global base_path_list, prefix_list

    base_path = os.path.join(DATA_PATH, "DeepQ-Synth-Hand/data")

    """for synth data"""
    for i in range(10):
        base_path_list.append(os.path.join(base_path, "s00{}".format(i)))
        prefix_list.append("{}".format(i))

    """for real data"""

    base_path_list += [os.path.join(DATA_PATH, "DeepQ-Vivepaper/data/air"),
                       os.path.join(DATA_PATH, "DeepQ-Vivepaper/data/book")]
    prefix_list += [10, 11]


CLASS_TO_INT = {
    "L": 1,
    "R": 2
}


def get_obj(base_path):
    print(base_path)
    data_obj = dict()

    img_path = os.path.join(base_path, "img")
    label_path = os.path.join(base_path, "label")

    for filename in os.listdir(img_path):
        id_ = (filename.split(".")[0]).replace("img_", "")
        data_obj[id_] = dict()
        with open(os.path.join(label_path, "label_{}.json".format(id_))) as json_file:
            data_obj[id_]["label"] = json.load(json_file)["bbox"]
            data_obj[id_]["filename"] = filename
    return data_obj


def create_tf_example(config, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(config["filename"])), 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = config["filename"].encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for key, value in config["label"].items():
        xmins.append(value[0] / width if value[0] > 0 else 0)
        xmaxs.append(value[2] / width if value[2] < width else 1)
        ymins.append(value[1] / height if value[1] > 0 else 0)
        ymaxs.append(value[3] / height if value[3] < height else 1)
        classes_text.append(key.encode('utf8'))
        classes.append(CLASS_TO_INT[key])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def preprocessing():
    try:
        with open("./data_configs.json", "r") as fp:
            data_configs = json.load(fp)
    except:
        data_configs = dict([(prefix_list[i], get_obj(base_path_list[i])) for i in range(len(prefix_list))])
        with open("./data_configs.json", "w") as fp:
            json.dump(data_configs, fp)

    if not os.path.exists(TRAIN_PATH):
        os.makedirs(TRAIN_PATH)

    if not os.path.exists(EVAL_PATH):
        os.makedirs(EVAL_PATH)

    for prefix, data_obj in data_configs.items():
        train_output_path = os.path.join(TRAIN_PATH, "{}.tfrecord".format(prefix))
        eval_output_path = os.path.join(EVAL_PATH, "{}.tfrecord".format(prefix))

        print('Start creating the TFRecords: {} and {}'.format(train_output_path, eval_output_path))
        train_writer = tf.python_io.TFRecordWriter(train_output_path)
        eval_writer = tf.python_io.TFRecordWriter(eval_output_path)
        n_examples = len(list(data_obj.keys()))
        ctr = 0
        for id_, config in data_obj.items():
            path = os.path.join(base_path_list[int(prefix)], "img")
            tf_example = create_tf_example(config, path)
            if ctr < TRAIN_RATIO * n_examples:
                train_writer.write(tf_example.SerializeToString())
            else:
                eval_writer.write(tf_example.SerializeToString())
            ctr += 1

        train_writer.close()
        eval_writer.close()
        print('Successfully created the TFRecords: {}  and {}'.format(train_output_path, eval_output_path))


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help="parent path of data e.g. D://data//htc")
    parser.add_argument('--path_type', type=int, default=0, help='path_type')
    args = parser.parse_args()
    try:
        DATA_PATH = args.data_path
    except:
        print("Need data path!!")

    if args.path_type == 0:
        generate_paths_0()
    else:
        generate_paths_1()

    preprocessing()
