import pdb
import argparse
import os
import glob
import json
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str, help='The path to a folder which contains 2 folders "img/" and "label/"')
args = parser.parse_args()

img_paths = glob.glob(args.dir + '/img/*')

label_dir = args.dir + '/label/'
flipped_img_dir = args.dir + '/img_f/'
flipped_label_dir = args.dir + '/label_f/'

if not os.path.isdir(flipped_img_dir):
    os.mkdir(flipped_img_dir)
if not os.path.isdir(flipped_label_dir):
    os.mkdir(flipped_label_dir)

for p in img_paths:
    img = Image.open(p)
    width, height = img.size

    img_id = int(p.split('/')[-1].split('.')[0].split('_')[1])

    with open(label_dir + 'label_' + '{:05d}'.format(img_id) + '.json', 'r') as f:
        label = json.load(f)

    label_new = {'bbox': {}}

    for h in label['bbox']:
        x0, y0, x1, y1 = label['bbox'][h]
        hf = 'R' if h == 'L' else 'L'
        label_new['bbox'][hf] = [width - x1, y0, width - x0, y1]

    img.transpose(Image.FLIP_LEFT_RIGHT).save(
        flipped_img_dir + 'img_' + '{:05d}'.format(len(img_paths) + img_id) + '.png', 'PNG')

    with open(flipped_label_dir + 'label_' + '{:05d}'.format(len(img_paths) + img_id) + '.json', 'w') as f:
        json.dump(label_new, f)
