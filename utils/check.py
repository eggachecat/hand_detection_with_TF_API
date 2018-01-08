import os
import glob
import json

import numpy as np
from skimage import io as img_io
# import matplotlib.pyplot as plt


def check_folder(folder):
    """
    check folder exist, if not, create one
    :param str folder:
    :return:
    """
    if os.path.exists(folder):
        user_input = input('folder {} already exist, do you want to overrider? [y/n] '
                           .format(folder))
        print('')
        if user_input in ('n', 'no', 'N', 'No'):
            print('please reset your arguments')
            exit(0)
    else:
        print('create new folder {}'.format(folder))
        os.mkdir(folder)
        os.mkdir(folder + '/logs')


class Data(object):
    def __init__(self, config):
        """
        config example:
            'load_min': True,
            'min_size': 100,
            'real_hand_path': './data/DeepQ-Vivepaper/data/',
            'rel_image_type': '.png',
            'synth_hand_path': './data/DeepQ-Synth-Hand-01/data/',
            'synth_image_type': '.png'

        :param dict config:
        """
        self.synth_hand_path = config['synth_hand_path']    # type: str
        self.real_hand_path = config['real_hand_path']      # type: str
        self.synth_image_type = config['synth_image_type']  # type: str
        self.real_image_type = config['real_image_type']    # type: str
        self.load_min = config['load_min']                  # type: bool
        self.min_size = config['min_size']                  # type: int
        self.sub_img_dirs = [sub_dir + '/' for sub_dir in sorted(os.listdir(config['synth_hand_path']))]  # type: list

    def _load_images(self, img_file_dir):
        img_file_list = sorted(glob.glob(img_file_dir + '*' + self.synth_image_type))
        if self.load_min:
            img_file_list = img_file_list[:self.min_size]

        # # load image and do normalize
        imgs = [img_io.imread(img_file) / 255 for img_file in img_file_list]

        return np.asanyarray(imgs)

    def _load_labels(self, label_file_dir):
        label_file_list = sorted(glob.glob(label_file_dir + '*.json'))
        if self.load_min:
            label_file_list = label_file_list[:self.min_size]

        labels = [self._parse_label_file(label_file) for label_file in label_file_list]

        return list(map(self._pad_labels, labels))

    @staticmethod
    def _parse_label_file(fp):
        with open(fp, 'r') as f:
            label_dict = json.load(f)

        return label_dict

    @staticmethod
    def _pad_labels(label):
        """
        remove keypoint info and pad mis hand with 0
        """
        label = label['bbox']
        if 'R' not in label:
            label['R'] = [0, 0, 0, 0]

        if 'L' not in label:
            label['L'] = [0, 0, 0, 0]

        return label

    def data_generator(self, batch_size=100, do_shuffle=False):
        while True:
            for sub_dir in self.sub_img_dirs:
                # # load images and labels first
                synth_imgs = self._load_images(self.synth_hand_path + sub_dir + 'img/')
                synth_labels = self._load_labels(self.synth_hand_path + sub_dir + 'label/')

                data_len = synth_imgs.shape[0]
                n_batch = data_len // batch_size
                indices_pool = [(i * batch_size, (i + 1) * batch_size) for i in range(n_batch)]
                if do_shuffle:
                    indices_order = np.random.permutation(n_batch)
                else:
                    indices_order = np.arange(n_batch)

                for index in indices_order:
                    split_pair = indices_pool[index]
                    s, e = split_pair[0], split_pair[1]

                    yield synth_imgs[s:e], synth_labels[s:e]


if __name__ == "__main__":
    pass
