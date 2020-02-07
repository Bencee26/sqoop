from matplotlib import pyplot as plt
import numpy as np
import os
import h5py
import math
import argparse


def get_conditions(path):

    # extended_path = path + "training"
    conditions = [int(d) for d in os.listdir(path) if os.path.isdir(path+'/'+d) and (d[-1].isdigit() or d[-2:].isdigit())]
    conditions = sorted(conditions)
    return conditions


def plot_frequencies(path):

    conditions = get_conditions(path)

    num_plots = len(conditions)
    num_plot_rows = math.ceil(num_plots/2)

    set_types = ["training", "validation", "test"]

    for set_type in set_types:

        im_freqs, q_freqs, num_samples = get_frequencies(conditions, set_type)

        for i in range(len(conditions)):
            k = conditions[i]
            freqs = [v for _, v in im_freqs[k].items()]

            plt.subplot(2, num_plot_rows, i+1)
            plt.hist(freqs)
            plt.title(f'Condition {k}')
            l = min(freqs)
            h = max(freqs)
            a = (l + h)//2
            t = [l, a, h]
            plt.xticks(t, t)
            plt.tight_layout()
        plt.savefig(f'{path}/image_frequencies_{set_type}.png')
        plt.close()

        for i in range(len(conditions)):
            k = conditions[i]
            freqs = [v for _, v in q_freqs[k].items()]
            plt.subplot(2, num_plot_rows, i + 1)
            plt.hist(freqs)
            plt.title(f'Condition {k}')
            t = [min(freqs), max(freqs)]
            plt.xticks(t, t)
            plt.tight_layout()
        plt.savefig(f'{path}/question_frequencies_{set_type}.png')
        plt.close()

        # plotting samples sizes
        ks, nums = list(zip(*[(k, v) for k, v in num_samples.items()]))
        ks = [str(k) for k in ks]
        plt.bar(ks, nums)
        plt.tight_layout()
        plt.savefig(f'{path}/number_of_{set_type}_samples.png')
        plt.close()


def plot_label_distribution(path):

    conditions = get_conditions(path)

    num_plots = len(conditions) * 2
    num_plot_rows = 4
    num_plot_cols = math.ceil(num_plots / num_plot_rows)

    for set_type in ["training", "validation", "test"]:

        im_freqs, q_freqs, _ = get_frequencies(conditions, set_type)
        pos_im_freqs, pos_q_freqs = get_pos_label_frequencies(conditions, set_type)
        assert len(im_freqs) == len(pos_im_freqs)
        assert len(q_freqs) == len(pos_q_freqs)

        for i in range(len(conditions)):
            k = conditions[i]

            im_ratios = [pos_im_freqs[k][j] / im_freqs[k][j] for j in range(len(im_freqs[k]))]
            q_ratios = [pos_q_freqs[k][j] / q_freqs[k][j] for j in range(len(q_freqs[k]))]

            plt.subplot(num_plot_rows, num_plot_cols, i + 1)
            plt.hist(im_ratios, bins=np.linspace(0, 1, 30, endpoint=True))
            plt.title(f'Image label ratios {k}')
            plt.subplot(num_plot_rows, num_plot_cols, len(conditions) + i + 1)
            plt.hist(q_ratios, bins=np.linspace(0, 1, 30, endpoint=True))
            plt.title(f'Question label ratios {k}')
            plt.tight_layout()

        plt.savefig(f'{path}/label_distribution_{set_type}.png')
        plt.close()


def get_frequencies(conditions, set_type):

    im_freqs = {k: {} for k in conditions}
    q_freqs = {k: {} for k in conditions}

    num_samples = {k: {} for k in conditions}

    for k in conditions:
        try:
            with h5py.File(f'{path}/{k}/{set_type}/samples.h5', 'r') as hdf:
                samples = hdf.get(f'samples')[:]
        except IOError:
            with h5py.File(f'{path}/{set_type}/{k}/samples.h5', 'r') as hdf:
                samples = hdf.get(f'samples')[:]

        num_samples[k] = len(samples)
        im_count = np.bincount(samples[:, 0])
        im_f = {i: im_count[i] for i in range(len(im_count))}
        im_freqs[k] = im_f

        q_count = np.bincount(samples[:,1])
        q_f = {i: q_count[i] for i in range(len(q_count))}
        q_freqs[k] = q_f

    return im_freqs, q_freqs, num_samples


def get_pos_label_frequencies(conditions, set_type):

    pos_im_freqs = {}
    pos_q_freqs = {}

    for k in conditions:

        try:
            with h5py.File(f'{path}/{k}/{set_type}/samples.h5', 'r') as hdf:
                samples = hdf.get(f'samples')[:]
        except IOError:
            with h5py.File(f'{path}/{set_type}/{k}/samples.h5', 'r') as hdf:
                samples = hdf.get(f'samples')[:]

        num_ims, num_qs, _ = np.max(samples, axis=0)+1
        im_l_f = {i: 0 for i in range(num_ims)}
        q_l_f = {i: 0 for i in range(num_qs)}

        for i in range(samples.shape[0]):
            if samples[i, 2] == 1:
                im_l_f[samples[i, 0]] += 1
                q_l_f[samples[i, 1]] += 1

        pos_im_freqs[k] = im_l_f
        pos_q_freqs[k] = q_l_f

    return pos_im_freqs, pos_q_freqs


def plot_num_questions(path):
    conditions = get_conditions(path)
    set_types = ["training", "validation", "test"]
    for set_type in set_types:
        num_questions = {}
        for k in conditions:
            try:
                with h5py.File(f'{path}/{k}/{set_type}/questions.h5', 'r') as hdf:
                    questions = hdf.get(f'questions')[:]
                    num_questions[k] = len(questions)
            except IOError:
                with h5py.File(f'{path}/{set_type}/{k}/questions.h5', 'r') as hdf:
                    questions = hdf.get(f'questions')[:]
                    num_questions[k] = len(questions)
        ks, nums = list(zip(*[(k, v) for k, v in num_questions.items()]))
        ks = [str(k) for k in ks]
        plt.bar(ks, nums)
        plt.tight_layout()
        plt.title(f'number of questions in {set_type} set')
        plt.savefig(f'{path}/number_of_questions')
        plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    path = args.path

    plot_num_questions(path)
    plot_frequencies(path)
    plot_label_distribution(path)
