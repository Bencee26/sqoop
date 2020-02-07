from sqoop.metrics import *
from sqoop.dataloader import load_data
import argparse
import pickle
import torch
from sqoop.utils import get_model_config
from sqoop.save import save_metrics
import numpy as np


def get_representations(model_name, set_type):

    path = f'saved/{model_name}/{set_type}_sender_repr.pkl'
    sender_repr = []
    with open(path, 'rb') as file:
        while True:
            try:
                sender_repr.append(pickle.load(file))
            except EOFError:
                break

    path = f'saved/{model_name}/{set_type}_receiver_repr.pkl'
    receiver_repr = []
    with open(path, 'rb') as file:
        while True:
            try:
                receiver_repr.append(pickle.load(file))
            except EOFError:
                break


    path = f'saved/{model_name}/{set_type}_image_features.pkl'
    image_features = []
    with open(path, 'rb') as file:
        while True:
            try:
                image_features.append(pickle.load(file))
            except EOFError:
                break

    return sender_repr, receiver_repr, image_features


def get_messages(model_name, set_type):

    path = f'saved/{model_name}/{set_type}_messages.txt'
    message = []
    with open(path, 'r') as file:
        while True:
            # read a single line
            line = file.readline()
            if not line:
                break
            l = [int(n) for n in line.split()]
            m = torch.from_numpy(np.array(l))
            message.append(m)

    return torch.stack(message)


def calculate_rsa(model_name):

    sender_repr, receiver_repr, image_features = get_representations(model_name, "test")
    sender_repr = torch.stack(sender_repr).cpu().detach().numpy()
    receiver_repr = torch.stack(receiver_repr).cpu().detach().numpy()

    image_features = torch.stack(image_features).cpu().detach().numpy()

    messages = get_messages(model_name, "test").cpu().detach().numpy()

    config_dict = get_model_config(model_name)
    game_type = config_dict['game_type']
    k = config_dict['k']
    dataset = config_dict['dataset']
    scale_input = config_dict['scale_input']
    word2idx = config_dict['word2idx']

    _, ground_truth, _, _, _ = load_data(game_type, "test", word2idx, scale_input, dataset, k=k, discard_image=True)
    ground_truth = ground_truth.cpu()

    metrics = {}

    metrics['rsa_sr'], metrics['rsa_si'], metrics['rsa_ri'], metrics['rsa_mi'] = representation_similarity_analysis(image_features, ground_truth, messages, sender_repr, receiver_repr)

    save_metrics(model_name, metrics)

    return metrics['rsa_sr'], metrics['rsa_si'], metrics['rsa_ri'], metrics['rsa_mi']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str)
    args = parser.parse_args()

    rsa_sr, rsa_si, rsa_ri, topological_sim = calculate_rsa(args.modelname)

    print(f'rsa sender-receiver: {rsa_sr}')
    print(f'rsa sender-input: {rsa_si}')
    print(f'rsa receiver-input: {rsa_ri}')
    print(topological_sim)




