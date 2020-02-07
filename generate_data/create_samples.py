import random
from sqoop_helper import *
import h5py
from random import shuffle


def decide_label(char_pair, r, gt):
    gt = [list(o) for o in gt]
    if r == "right_of":
        return int(gt[0].index(char_pair[0]) > gt[0].index(char_pair[1]))
    elif r == "left_of":
        return int(gt[0].index(char_pair[0]) < gt[0].index(char_pair[1]))
    elif r == "below":
        return int(gt[1].index(char_pair[0]) > gt[1].index(char_pair[1]))
    elif r == "above":
        return int(gt[1].index(char_pair[0]) < gt[1].index(char_pair[1]))


def create_samples(name, conditions, num_ims, qs, ground_truth, set_size, set_type):

    all_samples = {}
    all_im_frequencies = {}

    for k in conditions:
        print(f'Set type {set_type}, Condition {k}')

        samples = []
        im_frequencies = {i: 0 for i in range(num_ims)}

        while len(samples) < set_size:

            remaining_q_idx = list(range(len(qs[k])))
            while remaining_q_idx and len(samples) < set_size:
                im_freqs_sorted = sorted(im_frequencies.items(), key=lambda x: x[1])
                im_idxs_sorted = [x[0] for x in im_freqs_sorted]

                current_q_idx = random.choice(remaining_q_idx)
                current_q = qs[k][current_q_idx]
                current_pair, r = to_ch_pair(current_q)

                remaining_q_idx.remove(current_q_idx)

                # We search for an image that has those two characters (starting from the least used image)
                for im_idx in im_idxs_sorted:
                    if all(ch in ground_truth[im_idx][0] for ch in current_pair):

                        label = decide_label(current_pair, r, ground_truth[im_idx])

                        samples.append((im_idx, current_q_idx, label))
                        im_frequencies[im_idx] += 1

                        break

                    if im_idx == im_idxs_sorted[-1]:
                        raise Exception("Didn't find a matching images")

        im_freqs_sorted = sorted(im_frequencies.items(), key=lambda x: x[1])
        if im_freqs_sorted[0][1] == 0:
            raise Exception('There has been an image that has no question assigned to it')
        all_samples[k] = samples
        all_im_frequencies[k] = im_frequencies
        # saving train samples
        with h5py.File(f'{name}/{set_type}/{k}/samples.h5', 'w') as hdf:
            hdf.create_dataset('samples', data=samples)


def create_all_possible_samples(conditions, num_ims, qs, ground_truth):

    all_samples = {}

    for k in conditions:

        samples = []
        remaining_q_idx = list(range(len(qs[k])))
        while remaining_q_idx:

            current_q_idx = random.choice(remaining_q_idx)
            current_q = qs[k][current_q_idx]
            current_pair, r = to_ch_pair(current_q)

            rels = []
            for i in range(current_q_idx-3, current_q_idx+4):
                if i != current_q_idx:
                    char_tuple, r_relative = to_ch_pair(qs[k][i])
                    if char_tuple == current_pair:
                        rels.append((i, r_relative))
                        if len(rels) == 3:
                            break
            assert (len(rels) == 3)

            remaining_q_idx.remove(current_q_idx)
            for rel in rels:
                idx = rel[0]
                remaining_q_idx.remove(idx)

            for im_idx in range(num_ims):
                # check if we can ask the question about the image
                if all(ch in ground_truth[im_idx][0] for ch in current_pair):

                    label = decide_label(current_pair, r, ground_truth[im_idx])

                    samples.append((im_idx, current_q_idx, label))
                    # we know that we can ask the question about its relatives as well
                    for rel in rels:
                        idx, relation = rel
                        label = decide_label(current_pair, relation, ground_truth[im_idx])
                        samples.append((im_idx, idx, label))

        shuffle(samples)
        print(f'for condition {k}, {len(samples)} samples has been created')
        all_samples[k] = samples

        return all_samples


def is_related(current_pair, q):

    char_tuple, r = to_ch_pair(q)
    return char_tuple == current_pair
