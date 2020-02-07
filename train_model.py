import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import torch
import torch.optim as optim
import torch.nn as nn
import time
import string

from sqoop.model import Sqoop_model
from sqoop.vocab import create_vocab
from sqoop.save import save_model, save_stats, save_hparams, make_dir
from sqoop.unittest_model import test_variable_change, test_loss
from sqoop.evaluate import evaluate
from sqoop.utils import get_set_size, write_parameters, name_model, get_next_idxs, get_num_samples, get_next_mem_idxs, write_model_config, get_data_config
from sqoop.argparse import parse_arguments
from plot.plot_training import plot_training
from plot.plot_gradients import plot_grad_flow
from plot.plot_parameters import plot_parameter_chart
from sqoop.dataloader import load_data, build_batch, get_batch_idxs


def main(**kwargs):

    game_type = args.game_type

    if game_type == "multimodal":
        k = int(args.k)
    else:
        k = None

    data_config = get_data_config(args.dataset)
    if data_config:
        s = data_config['s']
    else:
        s = list(string.ascii_uppercase)

    model_name = name_model(sys.argv, **vars(args))
    batch_size = args.batch_size
    idx2word, word2idx = create_vocab(s)

    num_chars_per_image = int(args.dataset[-1])

    config_dict = {'dataset': args.dataset, "k": k, 'game_type': args.game_type, "idx2word": idx2word,
                   'word2idx': word2idx, "scale_input": args.scale_input, 'num_chars_per_image': num_chars_per_image}

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if args.pretrained:
        state = torch.load('saved/' + args.pretrained + '.pt', map_location=device)
        epoch = state['epoch']
        model = state['model']
        optimizer = state['optimizer']
        total_num_iters = state['total_num_iters']

    else:
        epoch = 0
        total_num_iters = 0
        if args.use_ground_truth:
            if args.vocab_size != len(s)+1 or args.max_sentence_length != 10:
                raise ValueError("You can only use ground truths with sentence length 10 and vocab size equal to the"
                                 "size of all characters +1 (for the null-token)")
        model = Sqoop_model(model_name=model_name,
                            idx2word=idx2word,
                            game_type=game_type,
                            arch=args.arch,
                            use_pretrained_features=args.use_pretrained_features,
                            use_ground_truth=args.use_ground_truth,
                            train_from_symbolic=args.train_from_symbolic,
                            bottleneck=args.bottleneck,
                            num_chars_per_image=num_chars_per_image,
                            continuous_communication=args.continuous_comm,
                            batch_size=args.batch_size,
                            vocab_size=args.vocab_size,
                            max_sentence_length=args.max_sentence_length,
                            num_stem_layers=args.num_stem_layers,
                            message_embedding_dim=args.message_embedding_dim,
                            message_lstm_hidden_size=args.message_lstm_hidden_size,
                            encoder_lstm_hidden_size=args.encoder_lstm_hidden_size,
                            question_embedding_dim=args.question_embedding_dim,
                            question_rnn_hidden_size=args.question_rnn_hidden_size,
                            film_input_size=args.film_input_size,
                            film_channels=args.film_channels,
                            mlp_hidden_dim=args.mlp_hidden_dim,
                            dropout_prob=args.dropout_prob
                            )

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if device == 'cuda':
        model.cuda()

    # unit testing model
    if args.test_model:
        test_loss(model)
        test_variable_change(model)

    #saving hyperparameters and model settings in json file
    if not args.debug:
        make_dir(model_name)
        save_hparams(args, model, model_name)
        # write_parameters(model_name, model)
        # plot_parameter_chart(model_name, model)
        write_model_config(model_name, **config_dict)

    print('device: ', device)

    criterion = nn.CrossEntropyLoss().to(device)

    gradual_memory_load = False
    loaded_to_mem_counter = 0
    load_to_mem_idxs = None # by default we load all images to memory (it is required for multimodal games!)
    if game_type == "referential":
        num_all_samples = get_num_samples(args.dataset, game_type, 'training')
        if num_all_samples > args.num_ims_in_memory:
            gradual_memory_load = True
            load_to_mem_idxs = list(range(args.num_ims_in_memory))
            loaded_to_mem_counter = args.num_ims_in_memory

    all_images, all_ground_truths, all_questions, all_samples, all_labels = load_data(game_type, "training", word2idx,
                                                                                      args.scale_input, args.dataset, k,
                                                                                      args.check_data_loading,
                                                                                      load_to_mem_idxs)
    if game_type == "multimodal":
        num_train_samples_in_mem = len(all_samples)
    if game_type == "referential":
        num_train_samples_in_mem = len(all_labels)

    running_loss = 0.0
    running_accuracy = 0.0
    running_entropy = 0.0
    running_md = 0.0
    running_elapsed = 0.0
    best_validation_accurcay = 0.0
    spec = ""

    # training loop, we train until iter_limit is reached
    done = False
    while not done:

        sample_counter = 0
        while sample_counter != num_train_samples_in_mem:

            start_time = time.time()
            if total_num_iters == args.iter_limit:
                spec = "final"
                done = True
                break

            sample_idxs, sample_counter = get_next_idxs(sample_counter, num_train_samples_in_mem, batch_size)
            train_images, train_ground_truth, train_questions, train_labels = build_batch(game_type, all_images,
                                                                                          all_ground_truths,
                                                                                          all_questions, all_samples,
                                                                                          sample_idxs, idx2word,
                                                                                          args.check_data_loading,
                                                                                          all_labels)

            model.train()
            optimizer.zero_grad()

            pred, comm_metrics = model(train_images, train_questions,
                                       cut_image_info=args.cut_image_info,
                                       cut_question_info=args.cut_question_info,
                                       ground_truth=train_ground_truth)

            accuracy = (train_labels == torch.argmax(pred, -1)).float().mean()
            loss = criterion(pred, train_labels)

            loss.backward()
            if args.plot_grad_flow:
                if total_num_iters % 500 == 0:
                    plot_grad_flow(model_name, model.named_parameters())
            optimizer.step()

            if args.bottleneck:
                running_entropy += comm_metrics['entropy']
                running_md += comm_metrics['md']
            running_loss += loss.item()
            running_accuracy += accuracy.item()
            total_num_iters += 1

            end_time = time.time()
            elapsed = end_time-start_time
            running_elapsed += elapsed

            if args.stat_save_interval != 0 and total_num_iters % args.stat_save_interval == 0:

                stats_for_save = {'epoch': epoch, 'total_num_iters': total_num_iters}

                avg_loss = round(running_loss/args.stat_save_interval, 3)
                avg_accuracy = round(running_accuracy/args.stat_save_interval, 4)
                avg_elapsed = round(running_elapsed/args.stat_save_interval, 2)
                stats_for_save['time/iter'] = avg_elapsed
                stats_for_save['loss'] = avg_loss
                stats_for_save['accuracy'] = avg_accuracy
                if args.bottleneck:
                    avg_entropy = running_entropy/args.stat_save_interval
                    stats_for_save['entropy'] = avg_entropy
                    avg_md = running_md/args.stat_save_interval
                    stats_for_save['md'] = avg_md

                print(f'Epoch: {epoch}, iteration: {total_num_iters}, '
                      f'Time per iteration: {avg_elapsed} sec, '
                      f'AvgTrainingLoss: {avg_loss}, '
                      f'AvgAccuracy: {avg_accuracy}')
                if not args.debug:
                    save_stats(model_name, 'training', stats_for_save)

                running_loss = 0.0
                running_accuracy = 0.0
                running_elapsed = 0.0
                if args.bottleneck:
                    running_entropy = 0.0
                    running_md = 0.0

            if args.validation_interval != 0 and total_num_iters % args.validation_interval == 0:
                validation_acc, avg_comm_metrics = evaluate(model, game_type, k, batch_size, word2idx, "validation",
                                                            args.dataset, args.scale_input)

                if validation_acc > best_validation_accurcay:

                    state = {'total_num_iters': total_num_iters, 'epoch': epoch, 'model': model, 'optimizer': optimizer,
                             'accuracy': round(validation_acc, 4)}
                    save_model(state, model_name)

                stats_for_save = {'epoch': epoch, 'total_num_iters': total_num_iters, 'accuracy': validation_acc}
                if avg_comm_metrics:
                    stats_for_save.update(avg_comm_metrics)
                save_stats(model_name, 'validation', stats_for_save)

                print(f'\n On validation: \n'
                      f'Epoch: {epoch}, iteration: {total_num_iters}, '
                      f'Validation Accuracy: {validation_acc} \n')

                plot_training(model_name, args.plot_loss, args.bottleneck)

        # we are done with the current samples in the memory
        if gradual_memory_load:

            loaded_to_mem_counter, load_to_mem_idxs, epoch_end = get_next_mem_idxs(num_all_samples,
                                                                                   loaded_to_mem_counter,
                                                                                   args.num_ims_in_memory)
            if epoch_end:
                epoch += 1

            all_images, all_ground_truths, all_questions, all_samples, all_labels = load_data(game_type, "training",
                                                                                              word2idx,
                                                                                              args.scale_input,
                                                                                              args.dataset, k,
                                                                                              args.check_data_loading,
                                                                                              load_to_mem_idxs)

        else:
            epoch += 1

    save_model(state, model_name, spec)


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))