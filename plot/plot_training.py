import matplotlib.pyplot as plt
import csv
import argparse
import numpy as np


def plot_training(modelname, plot_loss, plot_communication):

    header = {}
    stats = {}
    path = f'saved/{modelname}'
    for set_type in ["training", "validation"]:
        stats_array = []
        with open(f'{path}/{set_type}_stats.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            header[set_type] = next(csv_reader)
            for line in csv_reader:
                l = [float(stat) for stat in line]
                stats_array.append(l)

        stats_array = np.array(stats_array)
        stats[set_type] = {}
        for i in range(len(header[set_type])):
            stats[set_type][header[set_type][i]] = stats_array[:, i]

    # plotting

    for set_type in ['training', 'validation']:
        plt.plot(stats[set_type]['total_num_iters'], stats[set_type]['accuracy'], label=f'{set_type} accuracy')
    if plot_loss:
        plt.plot(stats['training']['total_num_iters'], stats['training']['loss'], label='training loss')
    plt.legend()
    plt.title('accuracy')
    plt.savefig(f'{path}/accuracies.png')
    plt.close()

    if plot_communication:
        # plot entropy
        for set_type in ['training', 'validation']:
            plt.plot(stats[set_type]['total_num_iters'], stats[set_type]['entropy'], label=f'{set_type} entropy')
        plt.legend()
        plt.title('entropy')
        plt.savefig(f'{path}/entropy.png')
        plt.close()

        # plot message distinctness
        for set_type in ['training', 'validation']:
            plt.plot(stats[set_type]['total_num_iters'], stats[set_type]['md'], label=f'{set_type} message distinctness')

        plt.legend()
        plt.title('message distinctness')
        plt.savefig(f'{path}/message_distinctness.png')
        plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, help="name of the model")
    parser.add_argument("--plot_loss", type=bool, default=True)
    parser.add_argument("--plot_communication", type=bool, default=True)
    args = parser.parse_args()

    plot_training(args.modelname, args.plot_loss, args.plot_communication)