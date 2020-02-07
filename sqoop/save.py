import torch
import os
import csv
import json


def make_dir(model_name):
    if not os.path.isdir('saved'):
        os.mkdir('saved')
    if not os.path.isdir(f'saved/{model_name}'):
        os.mkdir(f'saved/{model_name}')


def save_model(state, model_name, spec=None):
    if not os.path.isdir(f'saved/{model_name}'):
        make_dir(model_name)
    if not spec:
        torch.save(state, f'saved/{model_name}/best_model.pt')
    else:
        torch.save(state, f'saved/{model_name}/final_model.pt')


def save_stats(model_name, set_type, stats):
    path = f'saved/{model_name}/{set_type}_stats.csv'
    if not os.path.isfile(path):
        fieldnames = []
        for names, _ in stats.items():
            fieldnames.append(names)
        with open(path, 'w') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()

    with open(path, 'a') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        stats_list = []
        for _, stat in stats.items():
            stats_list.append(stat)
        csv_writer.writerow(stats_list)


def save_diagnostic(model_name, feature_type, stats):

    path = f'saved/{model_name}/diagnostic_from_{feature_type}.csv'
    if not os.path.isfile(path):
        fieldnames = []
        for names, _ in stats.items():
            fieldnames.append(names)
        with open(path, 'w') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()

    with open(path, 'a') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        stats_list = []
        for _, stat in stats.items():
            stats_list.append(stat)
        csv_writer.writerow(stats_list)

def save_hparams(args, model, model_name):
    hparams = {}
    for k, v in args.__dict__.items():
        hparams[k] = v
    for k, v in vars(model).items():
        if isinstance(v, int) or isinstance(v, float) or isinstance(v, bool):
            hparams[k] = v
    num_model_params = sum(p.numel()for p in model.parameters())
    hparams["num_model_params"] = num_model_params

    with open(f'saved/{model_name}/hparams.txt', 'a') as outfile:
        json.dump(hparams, outfile, indent=2)


def save_message(msg, model_name):
    path = f'saved/{model_name}/messages.txt'
    with open(path, 'a') as file:
        for i in range(len(msg)):
            m = [str(n) for n in msg[i].cpu().numpy()]
            m = " ".join(m[1:])
            file.write(m+'\n')


def save_ground_truths(gt, idx2word, model_name):

    path = f'saved/{model_name}/ground_truth.txt'
    with open(path, 'a') as file:
        for i in range(len(gt)):
            ground_string = [idx2word[n] for n in gt[i].cpu().numpy()]
            ground_string = " ".join(ground_string)
            file.write(ground_string+'\n')


def save_results(test_acc, model_name):
    path = f'saved/{model_name}/test_accuracy.txt'
    with open(path, 'w') as f:
        f.write(f'Test accuracy: {test_acc}')


def save_metrics(model_name, metrics):
    path = f'saved/{model_name}/metrics.csv'
    if not os.path.isfile(path):
        fieldnames = []
        for names, _ in metrics.items():
            fieldnames.append(names)
        with open(path, 'w') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()

    with open(path, 'a') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        stats_list = []
        for _, stat in metrics.items():
            stats_list.append(stat)
        csv_writer.writerow(stats_list)