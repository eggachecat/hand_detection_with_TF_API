# -*- coding: utf-8 -*-
import pdb
import yaml
import argparse

import matplotlib.pyplot as plt

from utils import check


def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL Final Project")
    parser.add_argument('-c', '--continue_train', action='store_true',
                        help='continue last time training')
    parser.add_argument('-m', '--model_path', type=str, default='./exp/exp_0/',
                        help='experiment path to save model')
    parser.add_argument('--config_file', type=str, default='./configs/setting.yml',
                        help='experiment path to save model')

    args = parser.parse_args()
    return args


def train(args):
    if args.continue_train:
        print("continue training")
        with open(args.model_path + 'setting.yml', 'r') as set_file:
            setting = yaml.load(set_file)

    else:
        print("training new model")
        with open(args.config_file, 'r') as set_file:
            setting = yaml.load(set_file)

        check.check_folder(args.model_path)
        with open(args.model_path + 'setting.yml', 'w') as out_set_file:
            yaml.dump(setting, out_set_file, default_flow_style=False)

    config = setting['config']
    hp = setting['hp']

    # # ===== get training data =====
    data = check.Data(config)
    data_gen = data.data_generator(batch_size=hp['batch_size'])

    for epoch in range(hp['epochs']):
        batch_imgs, batch_labels = next(data_gen)
        pdb.set_trace()


if __name__ == "__main__":
    my_args = parse()
    train(my_args)
