import argparse
import string
from PIL import ImageFont
import numpy as np
from random import shuffle
from sqoop_helper import create_char_tuples, mkdirs
from generate_images import generate_im_from_gt
import h5py
import json


def generate_referential_data(**kwargs):

    name = f'{args.training_size}_{args.num_chars_per_image}'
    full_name = f'data/referential/{name}'
    mkdirs(full_name)

    num_chars_per_image = args.num_chars_per_image
    im_size = 64

    font_size_dict = {2: 25, 3: 20, 4: 16, 5: 12}
    font_size = font_size_dict[num_chars_per_image]
    background_color = (100, 100, 100)
    letter_color = (100, 200, 0)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerif.ttf", font_size, encoding="unic")
    except:
        font = ImageFont.load_default()

    s = list(string.ascii_uppercase)[:args.num_chars_total]

    config_dict = {"s": s, "num_chars_per_image": num_chars_per_image, 'char_separation': args.char_separation,
                   'training_size': args.training_size, 'validation_size': args.validation_size,
                   'test_size' : args.test_size, 'num_distractor_ims': args.num_distractor_ims}
    write_data_config(full_name, **config_dict)

    set_types = ["training", "validation", "test"]

    set_sizes = {"training": args.training_size, "validation": args.validation_size, "test": args.test_size}

    for set_type in set_types:

        ground_truths = []
        images = []
        labels = []
        for i in range(set_sizes[set_type]):

            if i % 100 == 0:
                print(f'{i} images created')

            chars = [s[idx] for idx in np.random.choice(len(s), num_chars_per_image, replace=False)]

            if args.distractor_type != "negative_example":
                shuffle(chars)
                left_right_order = chars.copy()
                shuffle(chars)
                top_down_order = chars.copy()
            else:
                left_right_order = chars
                top_down_order = chars

            gt_original = (left_right_order, top_down_order)

            im_original = generate_im_from_gt(gt_original, im_size, font_size, background_color, letter_color, font, args.char_separation)
            im_positive = generate_im_from_gt(gt_original, im_size, font_size, background_color, letter_color, font, args.char_separation)
            # im_original.show()
            # im_positive.show()
            sample = [np.array(im_original), np.array(im_positive)]
            all_gt = [gt_original, gt_original]

            for _ in range(args.num_distractor_ims):
                im_dist, gt_dist = create_distractor(gt_original, args.distractor_type, im_size, font_size,
                                                     background_color, letter_color, font, args.char_separation)
                sample.append(np.array(im_dist))
                all_gt.append(gt_dist)

            # shuffle examples, create labels
            sample = np.array(sample)
            perm = [0] + list(np.random.permutation(3) + 1)
            label = perm.index(1) - 1
            shuffled_sample = np.array(sample)[perm]
            shuffled_gt = np.array(all_gt)[perm]
            images.append(np.array(shuffled_sample))
            ground_truths.append(shuffled_gt)
            labels.append(label)

        with h5py.File(f'{full_name}/{set_type}/images.h5', 'w') as hdf:
            hdf.create_dataset('images', data=images)
            hdf.create_dataset('ground_truths', data=np.string_(ground_truths))
            hdf.create_dataset('labels', data=labels)


def write_data_config(full_name, **kwargs):
    j = json.dumps(kwargs)
    f = open(f'{full_name}/config.json', "w")
    f.write(j)
    f.close()


def create_distractor(gt_original, type, im_size, font_size, background_color, letter_color, font, char_separation):

    if type== "random":
        lr = gt_original[0].copy()
        while lr == gt_original[0]:
            shuffle(lr)

        td = gt_original[1].copy()
        while td == gt_original[1]:
            shuffle(td)

    # use this to generate negative example
    if type == "negative_example":
        lr = gt_original[0].copy()
        td = gt_original[1].copy()
        td[1], td[2] = td[2], td[1]

    if type == "position_swap":
        swap_dict = {}
        orig = gt_original[0]
        swap_list = orig.copy()
        while orig == swap_list:
            shuffle(swap_list)
        for i in range(len(orig)):
            swap_dict[orig[i]] = swap_list[i]

        lr = []
        for i in range(len(gt_original[0])):
            lr.append(swap_dict[gt_original[0][i]])

        td = []
        for i in range(len(gt_original[1])):
            td.append(swap_dict[gt_original[1][i]])

    gt_dist = (lr, td)
    im_dist = generate_im_from_gt(gt_dist, im_size, font_size, background_color, letter_color, font, char_separation)

    return im_dist, gt_dist


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--training_size", type=int)
    parser.add_argument("--validation_size", type=int)
    parser.add_argument("--test_size", type=int)
    parser.add_argument("--num_chars_per_image", type=int)
    parser.add_argument("--num_chars_total", type=int)
    parser.add_argument("--num_distractor_ims", type=int)
    parser.add_argument("--char_separation", type=int, default=5)
    parser.add_argument("--distractor_type", default="position_swap")
    args = parser.parse_args()

    generate_referential_data(**vars(args))