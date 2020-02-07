import numpy as np
from random import sample
import os


def create_char_tuples(S, k):

    train_tuples = []
    all_tuples = []

    for x in S:
        rest = S.copy()
        rest.remove(x)
        all_rhs = [rest[index] for index in np.random.choice(len(rest), k, replace=False)]
        all_tuples += [(x, y) for y in rest]
        train_tuples += [(x, y) for y in all_rhs]
    assert (len(train_tuples) == len(S) * k)
    assert (len(all_tuples) == len(S) * (len(S) - 1))

    if k != len(S)-1:
        test_tuples = list(set(all_tuples) - set(train_tuples))
    else:
        test_tuples = train_tuples.copy()

    num_test = len(test_tuples)
    random_idxs = sorted(sample(list(range(num_test)), num_test//2))

    valid_array = np.array(test_tuples)[random_idxs]
    valid_tuples = list([tuple(t) for t in valid_array])
    test_tuples = list(set(test_tuples) - set(valid_tuples))

    if k != len(S)-1:
        assert len(test_tuples) + len(valid_tuples) + len(train_tuples) == len(all_tuples)

    return train_tuples, valid_tuples, test_tuples



def to_question(ch_pair, r):
    return str(ch_pair[0] + " " + r + " " + ch_pair[1])


def to_ch_pair(q):
    lhs, r, rhs = q.split()
    char_pair = (lhs, rhs)
    return char_pair, r


def mkdirs(name, conditions=None):
    remaining_name = name
    name_parts = []
    while "/" in remaining_name:
        idx = remaining_name.find('/')
        name_parts.append(remaining_name[:idx])
        remaining_name = remaining_name[idx+1:]

    name_parts.append(remaining_name)

    concat_name = ""
    if name_parts[0] not in os.listdir():
        os.mkdir(name_parts[0])
    concat_name += name_parts[0] + '/'

    for i in range(1, len(name_parts)):
        if name_parts[i] not in os.listdir(concat_name):
            os.mkdir(f'{concat_name}{name_parts[i]}')
        concat_name += name_parts[i] + '/'

    if conditions:
        for k in conditions:
            if str(k) not in os.listdir(concat_name):
                os.mkdir(f'{concat_name}{k}')
            concat_name += str(k) + '/'

        for set_type in ["training", "validation", "test"]:
            if set_type not in os.listdir(concat_name):
                os.mkdir(f'{concat_name}{set_type}')
        # step one level up in the directory structure:
        concat_name.replace(f'{k}/', "")

    # If we don't have conditions
    else:
        for set_type in ["training", "validation", "test"]:
            if set_type not in os.listdir(concat_name):
                os.mkdir(f'{concat_name}{set_type}')