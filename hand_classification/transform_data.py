import json
import os
import skimage
import skimage.io
import skimage.transform
import scipy.misc
import argparse

DATA_PATH = ""
TRAIN_RATIO = 0.9

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


data_counter = 0


def create_classification(config, path, folder):
    global data_counter
    img = skimage.io.imread(os.path.join(path, '{}'.format(config["filename"])))

    height, width = img.shape[:2]
    for key, value in config["label"].items():

        xmins = value[0]
        xmaxs = value[2]
        ymins = value[1]
        ymaxs = value[3]

        if xmins < 0 or xmaxs > width or ymins < 0 or ymaxs > height:
            continue

        cropped = img[ymins:ymaxs, xmins:xmaxs]
        resized = skimage.transform.resize(cropped, (64, 64), mode="constant")

        scipy.misc.imsave(os.path.join(folder, "{}-{}.jpg".format(data_counter, key)), resized)
        data_counter += 1


def generate_classification_data():
    print("generate_classification_data")
    try:
        with open("./data_configs.json", "r") as fp:
            data_configs = json.load(fp)
    except:
        print("except")
        data_configs = dict([(prefix_list[i], get_obj(base_path_list[i])) for i in range(len(prefix_list))])
        with open("./data_configs.json", "w") as fp:
            json.dump(data_configs, fp)

    train_folder = "./data/train"
    test_folder = "./data/test"

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    for prefix, data_obj in data_configs.items():

        n_examples = len(list(data_obj.keys()))
        ctr = 0
        for id_, config in data_obj.items():
            path = os.path.join(base_path_list[int(prefix)], "img")
            if ctr < TRAIN_RATIO * n_examples:
                create_classification(config, path, train_folder)
            else:
                create_classification(config, path, test_folder)
            ctr += 1


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

    generate_classification_data()
