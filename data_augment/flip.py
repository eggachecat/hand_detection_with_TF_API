import pdb
import argparse
import os
import glob
import json
from PIL import Image, ImageEnhance
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str, help='The path to a folder which contains 2 folders "img/" and "label/"')
parser.add_argument('--flip', action='store_true',
                    help='augment with flip image')
parser.add_argument('--bright', action='store_true',
                    help='augment with brightness on image')

args = parser.parse_args()

img_paths = glob.glob(args.dir + '/img/*')

label_dir = args.dir + '/label/'

if args.flip:
    flipped_img_dir = args.dir + '/img_f/'
    flipped_label_dir = args.dir + '/label_f/'

    # # check and make new folder
    if not os.path.isdir(flipped_img_dir):
        os.mkdir(flipped_img_dir)
    if not os.path.isdir(flipped_label_dir):
        os.mkdir(flipped_label_dir)

if args.bright:
    bright_img_dir = args.dir + '/img_b/'
    bright_label_dir = args.dir + '/label_b/'

    if not os.path.isdir(bright_img_dir):
        os.mkdir(bright_img_dir)
    if not os.path.isdir(bright_label_dir):
        os.mkdir(bright_label_dir)

if not (args.flip or args.bright):
    raise ValueError('No augment type given')

pbar = tqdm(total=len(img_paths))
for p in img_paths:
    img = Image.open(p)
    width, height = img.size

    img_id = int(p.split('/')[-1].split('.')[0].split('_')[1])

    with open(label_dir + 'label_' + '{:05d}'.format(img_id) + '.json', 'r') as f:
        label = json.load(f)

    if args.flip:
        label_new = {'bbox': {}}
        for h in label['bbox']:
            x0, y0, x1, y1 = label['bbox'][h]
            hf = 'R' if h == 'L' else 'L'
            label_new['bbox'][hf] = [width - x1, y0, width - x0, y1]

        img.transpose(Image.FLIP_LEFT_RIGHT).save(
            flipped_img_dir + 'img_' + '{:05d}'.format(len(img_paths) + img_id) + '.png', 'PNG')
        with open(flipped_label_dir + 'label_' + '{:05d}'.format(len(img_paths) + img_id) + '.json', 'w') as f:
            json.dump(label_new, f)

    if args.bright:
        brightness = ImageEnhance.Brightness(img)
        bright_img1 = brightness.enhance(1.5)
        bright_img2 = brightness.enhance(2.0)
        darker_img = brightness.enhance(0.7)

        bright_img1.save(
            bright_img_dir + 'img_' + '{:05d}'.format(len(img_paths) + img_id) + '.png', 'PNG')
        with open(bright_label_dir + 'label_' + '{:05d}'.format(len(img_paths) + img_id) + '.json', 'w') as f:
            json.dump(label, f)

        bright_img2.save(
            bright_img_dir + 'img_' + '{:05d}'.format(2 * len(img_paths) + img_id) + '.png', 'PNG')
        with open(bright_label_dir + 'label_' + '{:05d}'.format(2 * len(img_paths) + img_id) + '.json', 'w') as f:
            json.dump(label, f)

        darker_img.save(
            bright_img_dir + 'img_' + '{:05d}'.format(3 * len(img_paths) + img_id) + '.png', 'PNG')
        with open(bright_label_dir + 'label_' + '{:05d}'.format(3 * len(img_paths) + img_id) + '.json', 'w') as f:
            json.dump(label, f)

    pbar.update(1)
