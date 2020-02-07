import torch
import statistics
from sqoop.dataloader import load_data, build_batch, get_batch_idxs
from sqoop.utils import get_next_idxs, get_num_samples, get_next_mem_idxs
from sqoop.metrics import representation_similarity_analysis
import pickle


def save_feature(feature_to_save, set_type, model, comm_info):
    feature = comm_info[feature_to_save]
    path = f'saved/{model.model_name}/{set_type}_{feature_to_save}.pkl'
    with open(path, 'ab') as file:
        for i in range(len(feature)):
            pickle.dump(feature[i], file)


def evaluate(model, game_type, k, batch_size, word2idx, set_type, dataset, scale_input,
             save_messages=False, feature_to_save=None, num_ims_in_memory=500):

    model.eval()
    if feature_to_save:
        save_features = True
    else:
        save_features = False

    num_all_samples = get_num_samples(dataset, game_type, set_type)

    # gradual memory loading
    loaded_to_mem_counter = 0
    load_to_mem_idxs = None
    if num_all_samples > num_ims_in_memory:
        gradual_memory_load = True
        load_to_mem_idxs = list(range(num_ims_in_memory))
        loaded_to_mem_counter = num_ims_in_memory
    else:
        gradual_memory_load = False
        num_ims_in_memory = num_all_samples

    all_images, all_ground_truth, all_questions, all_samples, all_labels = load_data(game_type, set_type, word2idx,
                                                                                     scale_input, dataset, k,
                                                                                     idxs=load_to_mem_idxs)

    running_accuracy = []
    avg_comm_metrics = {}
    if model.bottleneck:
        running_entropy = []
        running_md = []

    done = False

    while not done:

        sample_counter = 0
        while sample_counter != num_ims_in_memory:

            sample_idxs, sample_counter = get_next_idxs(sample_counter, num_ims_in_memory, batch_size)
            images, ground_truth, questions, labels = build_batch(game_type, all_images, all_ground_truth, all_questions, all_samples, sample_idxs, all_labels=all_labels)

            pred, comm_info = model(images, questions, ground_truth=ground_truth, save_messages=save_messages, save_features=save_features)

            accuracy = torch.mean((labels == torch.argmax(pred, -1)).float()).item()
            running_accuracy.append(accuracy)

            if model.bottleneck:
                running_entropy.append(comm_info['entropy'])
                running_md.append(comm_info['md'])

            if save_messages:
                msg = comm_info['message']
                path = f'saved/{model.model_name}/{set_type}_messages.txt'
                with open(path, 'a') as file:
                    for i in range(len(msg)):
                        m = [str(n) for n in msg[i].cpu().numpy()]
                        m = " ".join(m[1:])
                        file.write(m + '\n')

            if feature_to_save:
                if feature_to_save == "all":
                    feats_to_save = ['image_features', 'sender_repr', 'receiver_repr']
                else:
                    feats_to_save = [feature_to_save]
                for feat in feats_to_save:
                    save_feature(feat, set_type, model, comm_info)

            if sample_counter == num_ims_in_memory and gradual_memory_load:
                loaded_to_mem_counter, load_to_mem_idxs, epoch_end = get_next_mem_idxs(num_all_samples,
                                                                                       loaded_to_mem_counter,
                                                                                       num_ims_in_memory)

                all_images, all_ground_truths, all_questions, all_samples, all_labels = load_data(game_type, set_type,
                                                                                                  word2idx, scale_input,
                                                                                                  dataset, k, False,
                                                                                                  load_to_mem_idxs)
                if epoch_end:
                    done = True

            if not gradual_memory_load:
                done=True

    avg_accuracy = statistics.mean(running_accuracy)
    if model.bottleneck:
        avg_comm_metrics['entropy'] = statistics.mean(running_entropy)
        avg_comm_metrics['md'] = statistics.mean(running_md)

    return avg_accuracy, avg_comm_metrics

