import h5py
import numpy as np
import torch
from sqoop.scale_input import scale_input
import random
from PIL import Image


def load_data(game_type, set_type, word2idx, scale_images, dataset, k=None, check_data_loading=False, idxs=None, discard_image=False):

    assert set_type == "test" or set_type == "training" or set_type == "validation"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # load images
    is_filtered = ("filtered" in dataset)
    if is_filtered:
        pos = dataset.find("filtered")
        filter_name = dataset[pos-1:]
        dataset = dataset[:pos-1]

    add_spec = ""
    if k:
        add_spec = f'{k}/'

    with h5py.File(f'data/{game_type}/{dataset}/{add_spec}{set_type}/images.h5', 'r') as hdf:
        gts = hdf.get('ground_truths')[:]
        if not idxs:
            idxs = list(range(len(gts)))
        gts = hdf.get('ground_truths')[idxs]

        if discard_image:
            images = None
            labels = None
        else:
            images = hdf.get('images')[idxs]
            if game_type == "referential":
                labels = hdf.get('labels')[idxs]
                labels = torch.from_numpy(labels).to(device)

            images = torch.from_numpy(images)

    if not check_data_loading and not discard_image:
        images = images.type('torch.FloatTensor').to(device)
        if game_type == "multimodal":
            images = images.permute(0, 3, 1, 2)
            # normalize images
            if scale_images:
                images = scale_input(images)
        elif game_type == "referential":
            images = images.permute(0, 1, 4, 2, 3)
            orig_dims = images.shape
            if scale_images:
                images = scale_input(images.view(-1, orig_dims[2], orig_dims[3], orig_dims[4])).view(orig_dims)

    ground_truth = torch.zeros(gts.shape, dtype=torch.long).to(device)
    for i in range(gts.shape[0]):
        for j in range(gts.shape[1]):
            for l in range(gts.shape[2]):
                for k in range(gts.shape[3]):
                    ground_truth[i, j, l, k] = word2idx[gts[i, j, l, k].decode()]
    ground_truth = ground_truth.view(ground_truth.shape[0], ground_truth.shape[1], -1).to(device)

    if game_type == "multimodal":

        # load_questions
        try:
            with h5py.File(f'data/{game_type}/{dataset}/{k}/{set_type}/questions.h5', 'r') as hdf:
                qs = hdf.get(f'questions')[:]
        except IOError:
            with h5py.File(f'data/{game_type}/{dataset}/{set_type}/{k}/questions.h5', 'r') as hdf:
                qs = hdf.get(f'questions')[:]

        questions = torch.zeros((len(qs), 3), dtype=torch.long).to(device)
        for i in range(len(qs)):
            q = qs[i].decode()
            q = q.split()
            for j in range(len(q)):
                questions[i, j] = word2idx[q[j]]

        # labels are included in the samples
        labels = None

        # loading samples
        if is_filtered:
            dataset = dataset + filter_name

        try:
            with h5py.File(f'data/{game_type}/{dataset}/{k}/{set_type}/samples.h5', 'r') as hdf:
                samples = hdf.get(f'samples')[:]
                samples = torch.from_numpy(samples).type(torch.long).to(device)
        except IOError:
            with h5py.File(f'data/{game_type}/{dataset}/{set_type}/{k}/samples.h5', 'r') as hdf:
                samples = hdf.get(f'samples')[:]
                samples = torch.from_numpy(samples).type(torch.long).to(device)
        samples = samples[torch.randperm(len(samples))]

    elif game_type == "referential":
        questions = None
        samples = None

    # if its multimodal, it returns images, gts, questions and samples
    # if its referential it returns images and labels


    return images, ground_truth, questions, samples, labels


def build_batch(game_type, all_images, all_ground_truths, all_questions, all_samples, sample_idxs, idx2word=None, check_data_loading=False, all_labels=None):

    if game_type == "multimodal":
        batch_images = []
        batch_ground_truths = []
        batch_questions = []
        batch_labels = []

        for i in sample_idxs:
            im_idx, q_idx, label = all_samples[i]
            batch_images.append(all_images[im_idx])
            batch_ground_truths.append(all_ground_truths[im_idx])
            batch_questions.append(all_questions[q_idx])
            batch_labels.append(label)

        batch_images = torch.stack(batch_images)
        batch_ground_truths = torch.stack(batch_ground_truths)
        batch_questions = torch.stack(batch_questions)
        batch_labels = torch.stack(batch_labels)

        if check_data_loading:
            batch_size = batch_images.shape[0]
            idx = random.randint(0, batch_size)
            print(f'label: {batch_labels[idx].item()}')

            question = batch_questions[idx]
            q = ""
            for i in question:
                q += idx2word[i.item()]
                q += " "
            print(f'question: {q}')

            gt = batch_ground_truths[idx]
            ground = ""
            for char in gt:
                ground += idx2word[char.item()]
                ground += " "
            print(ground)

            im = batch_images[idx]
            img = Image.fromarray(im.numpy())
            img.show()

    if game_type == "referential":
        batch_images = all_images[sample_idxs]
        batch_ground_truths = all_ground_truths[sample_idxs]
        batch_labels = all_labels[sample_idxs]
        batch_questions = None

    return batch_images, batch_ground_truths, batch_questions, batch_labels


def get_batch_idxs(available_indices, batch_size):

    if batch_size < len(available_indices):
        random_indices = list(sorted(np.random.choice(available_indices, batch_size, replace=False)))
    else:
        random_indices = sorted(available_indices)
    available_indices = list(set(available_indices)-set(random_indices))

    return available_indices, random_indices


def tensor_to_question(tensor, idx2word):

    q = ""
    for idx in tensor:
        q += idx2word[idx.item()]
        q += " "
    q = q[:-1]

    return q


def load_diagnostic_batch(start_idx, batch_size, max_index, messages, ground_truths, num_available_chars, use_gt=False):
    if start_idx + batch_size < max_index:
        idxs = list(range(start_idx, start_idx+batch_size))
    else:
        idxs = list(range(start_idx, max_index))

    batch_gt = ground_truths[idxs]
    batch_message = messages[idxs]

    if use_gt:
        batch_message.fill_(0)
        batch_message[:, :batch_gt.shape[1]] = batch_gt

    batch_labels = ground_truth_to_label(batch_gt, num_available_chars)
    assert (len(batch_message) == len(batch_labels))

    return batch_message, batch_labels


def ground_truth_to_label(gt, num_available_chars):
    labels = torch.zeros(len(gt), num_available_chars).type(torch.FloatTensor)

    for i in range(len(gt)):
        for j in range(len(gt[0])):
            labels[i, gt[i, j]] = 1
    return labels
