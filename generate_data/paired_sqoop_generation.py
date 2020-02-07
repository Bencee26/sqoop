import random
import argparse
from sqoop_helper import create_char_tuples, mkdirs, to_question
import string
from generate_images import generate_im_from_gt
from random import shuffle
from PIL import ImageFont
import numpy as np
import h5py
from create_samples import create_all_possible_samples


def generate(num_training_ims, num_validation_ims, num_test_ims, num_chars_per_im, total_num_chars, k, char_separation):
    s = list(string.ascii_uppercase)[:total_num_chars]

    data_name = f'{num_chars_per_im}_{num_training_ims}_sep_{char_separation}'
    path = f'data/multimodal/{data_name}'
    mkdirs(path, conditions=[k])

    sizes = num_training_ims, num_validation_ims, num_test_ims

    #setting image characteristics
    im_size = 64
    font_size_dict = {2: 25, 3: 20, 4: 16, 5: 12}
    font_size = font_size_dict[num_chars_per_im]
    background_color = (100, 100, 100)
    letter_color = (100, 200, 0)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerif.ttf", font_size, encoding="unic")
    except:
        font = ImageFont.load_default()
    image_characteristics = [im_size, font_size, background_color, letter_color, font, char_separation]

    relations = ['left_of', 'right_of', 'above', 'below']
    set_types = ['training', 'validation', 'test']

    tuples = create_char_tuples(s, k)

    question_tuples = {set_types[i]: tuples[i] for i in range(len(set_types))}
    set_sizes = {set_types[i]: sizes[i] for i in range(len(set_types))}

    for set_type in set_types:
        questions = [to_question(ch_pair, r) for ch_pair in question_tuples[set_type] for r in relations]
        with h5py.File(f'{path}/{k}/{set_type}/questions.h5', 'w') as hdf:
            hdf.create_dataset('questions', data=np.string_(questions))

        images, ground_truths = paired_sqoop_generation(set_sizes[set_type], num_chars_per_im,
                                                        question_tuples[set_type], s, image_characteristics)

        with h5py.File(f'{path}/{k}/{set_type}/images.h5', 'w') as hdf:
            hdf.create_dataset('images', data=images)
            hdf.create_dataset('ground_truths', data=np.string_(ground_truths))

        questions_for_sample_creation = {k: questions}
        all_samples = create_all_possible_samples([k], len(images), questions_for_sample_creation, ground_truths)
        samples = all_samples[k]

        with h5py.File(f'{path}/{k}/{set_type}/samples.h5', 'w') as hdf:
            hdf.create_dataset('samples', data=samples)


def paired_sqoop_generation(total_num_ims, num_chars_per_im, question_tuples, s, image_characteristics):

    tuple_counts = [(t, 0) for t in question_tuples]
    ims = []
    gts = []
    while len(ims) != total_num_ims:

        assert len(tuple_counts) == len(question_tuples)

        if len(ims) % 100 == 0:
            print(f'{len(ims)} images generated')

        t1, c1 = tuple_counts[0]
        c1 += 1
        tuple_counts.pop(0)
        assert len(tuple_counts)+1 == len(question_tuples)

        if num_chars_per_im == 2:
            chars_on_image = list(t1)
            tuple_counts = insert_to_place(t1, c1, tuple_counts, 0)
        else:
            t2, c2, i = find_partner(t1, num_chars_per_im, tuple_counts)
            tuple_counts.pop(i)
            c2 += 1

            assert len(tuple_counts)+2 == len(question_tuples)
            tuple_counts = insert_to_place(t2, c2, tuple_counts, i-1)
            assert len(tuple_counts)+1 == len(question_tuples)
            tuple_counts = insert_to_place(t1, c1, tuple_counts, 0)
            assert len(tuple_counts) == len(question_tuples)

            # generate 2 ims to which both tuples can relate
            chars_on_image = list(set(list(t1) + list(t2)))

        if len(chars_on_image) < num_chars_per_im:
            necessary_chars = num_chars_per_im - len(chars_on_image)
            rest = set(s) - set(chars_on_image)
            for _ in range(necessary_chars):
                c = random.choice(list(rest))
                chars_on_image.append(c)
                rest = rest - set(c)

        # generate ground truths
        chars = chars_on_image
        shuffle(chars)
        left_right_order = chars.copy()
        shuffle(chars)
        top_down_order = chars.copy()

        gt = (left_right_order, top_down_order)

        im = generate_im_from_gt(gt, *image_characteristics)
        ims.append(np.array(im))
        gts.append(np.array(gt))

    return ims, gts


def insert_to_place(t, c, tuple_count, start_pos):
    for i in range(start_pos, len(tuple_count)):
        if tuple_count[i][1] > c:
            tuple_count = tuple_count[:i] + [(t, c)] + tuple_count[i:]
            break
        if i == (len(tuple_count)-1):
            tuple_count.append((t, c))

    return tuple_count


def find_partner(t1, num_chars_per_im, ordered_tuples):

    for i in range(len(ordered_tuples)):
        t, c = ordered_tuples[i]
        # check if we have can connect through an image
        num_chars_from_tuple = len(set(list(t1) + list(t)))
        if num_chars_from_tuple <= num_chars_per_im:
            break

    return t, c, i


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int)
    parser.add_argument("--num_training_ims", type=int)
    parser.add_argument("--num_validation_ims", type=int)
    parser.add_argument("--num_test_ims", type=int)
    parser.add_argument("--total_num_chars", type=int, default=26)
    parser.add_argument("--num_char_per_im", type=int)
    parser.add_argument("--char_separation", type=int, default=5)
    args = parser.parse_args()

    generate(args.num_training_ims, args.num_validation_ims, args.num_test_ims,
             args.num_char_per_im, args.total_num_chars, args.k, args.char_separation)
