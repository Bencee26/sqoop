import numpy as np
from PIL import Image
from PIL import ImageDraw
from random import shuffle
import random
import h5py


def generate_random_images(num_chars, name, num_ims, S, train_tuples, set_type, im_size, font_size, background_color, letter_color, font):

    ground_truth = []
    images = []
    for i in range(num_ims):

        if i % 100 == 0:
            print(f'{i} images created')

        possible_questions = 0
        while possible_questions < 1:
            chars = [S[idx] for idx in np.random.choice(len(S), num_chars, replace=False)]

            # check for all conditions that we can ask questions about at least 2 char-pairs
            for k, v in train_tuples.items():
                possible_questions = sum(tup[0] in chars and tup[1] in chars for tup in v)
                if possible_questions < 1:
                    break

        shuffle(chars)
        left_right_order = chars.copy()
        shuffle(chars)
        top_down_order = chars.copy()

        gt = (left_right_order, top_down_order)

        im = generate_im_from_gt(gt, im_size, font_size, background_color, letter_color, font, char_separation)

        ground_truth.append(np.array(gt))
        images.append(np.array(im))

        with h5py.File(f'{name}/{set_type}/images.h5', 'w') as hdf:
            hdf.create_dataset('images', data=images)
            hdf.create_dataset('ground_truths', data=np.string_(ground_truth))

    return ground_truth, images


def generate_im_from_gt(gt, im_size, letter_mask_size, background_color, letter_color, font, char_separation=5):

    background = np.ones((im_size, im_size, 3)) * np.array(background_color)
    image = Image.fromarray(background.astype('uint8'))
    poss_dict = gen_positions_from_gt(gt, im_size, letter_mask_size, char_separation)

    im = draw_letters(image, poss_dict, letter_color, font)

    # #to double check image generation
    # im.show()
    # print(gt)

    return im


def gen_positions_from_gt(gt, im_size, letter_mask_size, char_separation):

    num_chars = len(gt[0])
    poss_dict = {}
    tries = 0
    while not poss_dict:

        # we samples x and y coordinates in sorted order with at least char_separation spaces between them
        max_coordinate = im_size - letter_mask_size
        sample_num = max_coordinate - (num_chars-1)*(char_separation-1)
        x_coords = [(char_separation-1)*i + x for i, x in enumerate(sorted(random.sample(range(sample_num), num_chars)))]
        y_coords = [(char_separation-1)*i + x for i, x in enumerate(sorted(random.sample(range(sample_num), num_chars)))]

        poss_dict = {}
        for char in gt[0]:
            idx = [order.index(char) for order in gt]
            poss_dict[char] = (x_coords[idx[0]], y_coords[idx[1]])

        if any(is_overlapping(pos1, pos2, letter_mask_size) for pos1 in list(poss_dict.values())
                                                               for pos2 in list(poss_dict.values())
                                                               if pos1!=pos2):
            tries += 1
            poss_dict = {}

    return poss_dict


def generate_positions(existing_positions, nr, im_size, letter_mask_size):
    # generates letter positions so there is no overlap between the letter
    poss = existing_positions
    for i in range(nr):
        pos_cand = generate_candidate_pos(im_size, letter_mask_size)
        while any(is_overlapping(pos_cand, p, letter_mask_size) for p in poss) or has_equal_coordinate(pos_cand,
                                                                                                       poss):
            pos_cand = generate_candidate_pos(im_size, letter_mask_size)
        poss.append(pos_cand)
    return poss


def generate_candidate_pos(im_size, letter_mask_size):
    p = (np.random.choice(range(im_size - letter_mask_size)), np.random.choice(range(im_size - letter_mask_size)))
    return p


def is_overlapping(pos1, pos2, letter_mask_size):
    return all(np.abs(np.array(pos1) - np.array(pos2)) < np.array([letter_mask_size, letter_mask_size]))


def has_equal_coordinate(pos_cand, poss):
    return any(pos_cand[0] == p[0] for p in poss) or any(pos_cand[1] == p[1] for p in poss)


def draw_letters(image, pos_dict, letter_color, font):
    for letter, position in pos_dict.items():
        draw = ImageDraw.Draw(image)
        draw.text(position, str(letter), letter_color, font=font)
    return image