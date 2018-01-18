import argparse
import os
import glob
import json
from PIL import Image
import numpy as np


def coord_round(coord, mn, mx):
    if coord < mn: return mn
    if coord > mx: return mx
    return coord


parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str, help='The path to a folder which contains 2 folders "img/" and "label/"')
parser.add_argument('--new_pic_num', type=int, default=5)
parser.add_argument('--stddev', type=float, default=3)
args = parser.parse_args()

img_paths = glob.glob(args.dir + '/img/*')
label_dir = args.dir + '/label/'
noisy_label_dir = args.dir + '/label_n/'

if not os.path.isdir(noisy_label_dir):
    os.mkdir(noisy_label_dir)

for p in img_paths:
    img = Image.open(p)
    width, height = img.size
    img_id = p.split('.')[0].split('_')[1]

    with open(label_dir + 'label_' + img_id + '.json', 'r') as f:
        label = json.load(f)

    for i in range(args.new_pic_num):
        label_new = {'bbox': {}}
        for h in label['bbox']:
            old_coord = np.array(label['bbox'][h])
            while True:
                new_coord = old_coord + np.random.normal(scale=args.stddev, size=4)
                new_coord = np.around(new_coord).astype(np.int64)
                new_coord[0] = coord_round(new_coord[0], 0, width - 1)
                new_coord[1] = coord_round(new_coord[1], 0, height - 1)
                new_coord[2] = coord_round(new_coord[2], 0, width - 1)
                new_coord[3] = coord_round(new_coord[3], 0, height - 1)
                if new_coord[0] < new_coord[2] and new_coord[1] < new_coord[3]:
                    break
            label_new['bbox'][h] = [int(n) for n in new_coord]

        with open(noisy_label_dir + 'label_' + img_id + '_' + str(i) + '.json', 'w') as f:
            json.dump(label_new, f)
