import h5py
import re
import datetime
import torch
import string
import os
import json


def get_set_size(k, set_type, use_pretrained_features=True):

    if not use_pretrained_features and set_type == 'training':
        with h5py.File(f'data/{k}/train_qs_{k}.h5', 'r') as hdf:
            qs = hdf.get('training_qs')[:]
            set_size = qs.shape[0]

    else:
        with h5py.File(f'data/{k}/{set_type}/questions.h5', 'r') as hdf:
            qs = hdf.get('questions')[:]
            set_size = qs.shape[0]

    return set_size


def name_model(cmd_args, **kwargs):

    # naming file
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    cmd_name = ""
    for i in range(2, len(cmd_args)):
        cmd_name += cmd_args[i]
        cmd_name += "_"
    cmd_name = cmd_name.replace('/', '_')

    model_name = f'{timestamp}_{cmd_name}--k_{kwargs["k"]}'
    return model_name


def get_conditions(path):
    extended_path = path + "training"
    conditions = [int(d) for d in os.listdir(extended_path) if os.path.isdir(extended_path+'/'+d) and (d[-1].isdigit() or d[-2:].isdigit())]
    conditions = sorted(conditions)

    return conditions


def write_parameters(model_name, model):

    path = f'saved/{model_name}'
    with open(f'{path}/parameters_sizes.txt', 'w') as file:
        file.write('-\n')
        total_params = sum([p[1].numel() for p in model.named_parameters()])
        first_line = f'Total number of parameters: {total_params}\n'
        file.write(first_line)
        file.write('-\n')

        stem_params = sum([p.numel() for p in model.stem_conv.parameters()])
        file.write(f'Number of stem parameters: {stem_params}\n')

        bottleneck_params = 0
        if model.bottleneck:
            bottleneck_in_fc_params = sum([p.numel() for p in model.bottleneck_in_fc.parameters()])
            bottleneck_params += bottleneck_in_fc_params
            lstm_cell_params = sum([p.numel() for p in model.lstm_cell.parameters()])
            bottleneck_params += lstm_cell_params
            hidden2vocab_params = sum([p.numel() for p in model.hidden2vocab.parameters()])
            bottleneck_params += hidden2vocab_params
            message_embedding_params = model.message_embedding.shape[0] * model.message_embedding.shape[1]
            bottleneck_params += message_embedding_params
            message_encoder_lstm_params = sum([p.numel() for p in model.message_encoder_lstm.parameters()])
            bottleneck_params += message_encoder_lstm_params
            aff_transform_params = sum([p.numel() for p in model.aff_transform.parameters()])
            bottleneck_params += aff_transform_params

        file.write(f'Number of bottleneck parameters: {bottleneck_params}\n')

        question_params = 0
        question_embedding_params = sum([p.numel() for p in model.question_embedding.parameters()])
        question_params += question_embedding_params
        question_rnn_params = sum([p.numel() for p in model.question_rnn.parameters()])
        question_params += question_rnn_params

        file.write(f'Number of question parameters: {question_params}\n')

        film0_params = sum([p.numel() for p in model.FiLM_0.parameters()])
        film1_params = sum([p.numel() for p in model.FiLM_1.parameters()])
        film_params = film0_params + film1_params

        file.write(f'Number of FiLM parameters: {film_params}\n')

        mlp_params = sum([p.numel() for p in model.mlp.parameters()])

        file.write(f'Number of mlp parameters: {mlp_params}\n')
        file.write('-\n')

    assert(total_params == stem_params + bottleneck_params + question_params + film_params + mlp_params)
    create_param_table(model_name)


def create_param_table(model_name):

    path = f'saved/{model_name}'

    with open(f'{path}/parameters_sizes.txt', 'r') as file:
        data = file.readlines()
    longest_line = max([len(line) for line in data])

    for i in range(len(data)):
        if i != 0 and i != len(data)-1:
            data[i] = '|' + data[i]
        else:
            data[i] = " " + data[i]

        if data[i][1] == "-":
            data[i] = data[i][:-1] + ('-' * (longest_line-2))
        else:
            m = re.search("\d", data[i])
            idx = m.start()
            data[i] = data[i][:idx] + " " * (longest_line-len(data[i])+1) + data[i][idx:-1]

        if i != 0 and i != len(data) - 1:
            data[i] = data[i] + "|\n"
        else:
            data[i] = data[i] + "\n"



    with open(f'{path}/parameters_sizes.txt', 'w') as file:
        line = file.writelines(data)


def batch_2_onehot(batch_tensor, message_length, vocab_size):

    batch_size = batch_tensor.shape[0]
    onehot = torch.zeros((batch_size, message_length + 1, vocab_size))

    for i in range(batch_size):
        # we always start with the null token
        onehot[i, 0, 0] = 1
        for j in range(1, message_length+1):
            onehot[i, j, batch_tensor[i, j-1]] = 1

    return onehot


def get_next_idxs(sample_counter, num_samples, batch_size):

    if sample_counter + batch_size < num_samples:
        sample_indices = list(range(sample_counter, sample_counter+batch_size))
        sample_counter += batch_size
    else:
        sample_indices = list(range(sample_counter, num_samples))
        sample_counter = num_samples

    return sample_indices, sample_counter


def get_num_samples(dataset, game_type, set_type, k=None):

    add_spec = ""
    if k:
        add_spec = "k/"

    if game_type == "multimodal":
        with h5py.File(f'data/{game_type}/{dataset}/{add_spec}{set_type}/samples.h5', 'r') as hdf:
            s = hdf.get('samples')[:]
            num = len(s)

    if game_type == "referential":
        with h5py.File(f'data/{game_type}/{dataset}/{add_spec}{set_type}/images.h5', 'r') as hdf:
            # we load ground truths since it is faster to load but has the same number of elements as pictures
            gts = hdf.get('ground_truths')[:]
            num = len(gts)

    return num


def get_next_mem_idxs(num_all_samples, loaded_to_mem_counter, num_ims_in_memory):

    new_epoch = False
    if loaded_to_mem_counter != num_all_samples:
        if loaded_to_mem_counter + num_ims_in_memory <= num_all_samples:
            load_to_mem_idxs = list(range(loaded_to_mem_counter, loaded_to_mem_counter + num_ims_in_memory))
            loaded_to_mem_counter = loaded_to_mem_counter + num_ims_in_memory
        else:
            load_to_mem_idxs = list(range(loaded_to_mem_counter, num_all_samples))
            loaded_to_mem_counter = num_all_samples

    else:
        # That means we loaded all samples, we exit the inner loop and start a new epoch
        loaded_to_mem_counter = num_ims_in_memory
        load_to_mem_idxs = list(range(loaded_to_mem_counter))
        new_epoch = True

    return loaded_to_mem_counter, load_to_mem_idxs, new_epoch


def write_model_config(model_name, **kwargs):

    j = json.dumps(kwargs)
    f = open(f'saved/{model_name}/config.json', "w")
    f.write(j)
    f.close()


def get_model_config(model_name):

    f = open(f'saved/{model_name}/config.json', "r")
    config_dict = json.load(f)
    return config_dict


def get_data_config(dataset):

    try:
        f = open(f'saved/{dataset}/config.json', "r")
        config_dict = json.load(f)
    except IOError:
        config_dict = {}
    return config_dict

