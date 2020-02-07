import numpy as np
import string
from PIL import ImageFont
import h5py
import argparse
from sqoop_helper import *
from create_samples import create_samples, create_all_possible_samples
from generate_images import generate_random_images

parser = argparse.ArgumentParser()
parser.add_argument("--create_all_possible_samples", default=True)
parser.add_argument("--training_set_size")
parser.add_argument("--validation_set_size")
parser.add_argument("--test_set_size")
parser.add_argument("--num_training_ims")
parser.add_argument("--num_validation_ims")
parser.add_argument("--num_test_ims")
parser.add_argument("--num_chars", type=int)
args = parser.parse_args()


path = f'data/multimodal/{args.num_chars}_{args.num_training_ims}'
num_ims = {"training": int(args.num_training_ims),
           "validation": int(args.num_validation_ims),
           "test": int(args.num_test_ims)}

# set_size is only used if we dont create all the possible combinations of samples
if not create_all_possible_samples:
    set_size = {"training": int(args.training_set_size),
                "validation": int(args.validation_set_size),
                "test": int(args.test_set_size)}


im_size = 64
font_size = 10
background_color = (100, 100, 100)
letter_color = (100, 200, 0)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerif.ttf", font_size, encoding="unic")
except:
    font = ImageFont.load_default()

S = list(string.ascii_uppercase)
relations = ["right_of", "left_of", "above", "below"]
conditions = [1, 2, 4, 8, 16, 25]

mkdirs(name, conditions)

set_types = ["training", "validation", "test"]

tuples = {}
qs = {}
for st in set_types:
    tuples[st] = {}
    qs[st] = {}

for k in conditions:
    train_tup, valid_tup, test_tup = create_char_tuples(S, k)
    tuples["training"][k] = train_tup
    tuples["validation"][k] = valid_tup
    tuples["test"][k] = test_tup

    qs["training"][k] = [to_question(ch_pair, r) for ch_pair in train_tup for r in relations]
    qs["validation"][k] = [to_question(ch_pair, r) for ch_pair in valid_tup for r in relations]
    qs["test"][k] = [to_question(ch_pair, r) for ch_pair in test_tup for r in relations]

    for st in set_types:
        with h5py.File(f'{path}/{st}/{k}/questions.h5', 'w') as hdf:
            hdf.create_dataset(f'questions', data=np.string_(qs[st][k]))

print('Questions created')

ground_truth = {}
images = {}
for st in set_types:
    ground_truth[st], images[st] = generate_random_images(args.num_chars, path, num_ims[st], S,
                                                          tuples[st], st, im_size, font_size, background_color,
                                                          letter_color, font)


print("images created")


for st in set_types:
    if create_all_possible_samples:
        all_samples = create_all_possible_samples(path, conditions, len(images[st]), qs[st], ground_truth[st], st)
    else:
        create_samples(name, conditions, len(images[st]), qs[st], ground_truth[st], set_size[st], st)
for k in conditions:
    samples = all_samples[k]
    with h5py.File(f'{path}/{st}/{k}/samples.h5', 'w') as hdf:
        hdf.create_dataset('samples', data=samples)

print("Samples created")







