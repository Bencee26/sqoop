import torch
import argparse
import os
import pickle
from sqoop.evaluate import evaluate
from sqoop.utils import get_model_config
from diagnostic_classifier import diagnostic_classify, loadsets_to_list
from calculate_metrics import  calculate_rsa


def save_and_diagnose(**kwargs):

    set_list = loadsets_to_list(args.load_training, args.load_test)

    # 1 deleting existing feature saving
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    state = torch.load(f'saved/{args.modelname}/best_model.pt', map_location=device)
    model = state['model']

    config_dict = get_model_config(args.modelname)
    game_type = config_dict['game_type']
    k = config_dict['k']
    dataset = config_dict['dataset']
    scale_input = config_dict['scale_input']
    word2idx = config_dict['word2idx']

    # cleaning files
    set_types_to_delete = ['training', 'test']
    feature_types = ['receiver_repr', 'sender_repr', 'image_features', 'messages']
    to_delete_1 = [f'{s}_{f}.pkl' for s in set_types_to_delete for f in feature_types]
    to_delete_2 = [f'diagnostic_from_{f}' for f in feature_types]
    to_delete_3 = ["metrics.csv"]
    to_delete = to_delete_1 + to_delete_2 + to_delete_3
    delete_path = f'saved/{model.model_name}/'
    for f in to_delete:
        if f in os.listdir(delete_path):
            os.remove(delete_path + f)

    print('model loading and deletion done')
    # 2: forward pass on training set, saving relevant feature

    if args.feature_type != "messages":
        for set_type in set_list:
            _, _ = evaluate(model, game_type, k, args.batch_size, word2idx, set_type, dataset, scale_input,
                            save_messages=False, feature_to_save=args.feature_type, num_ims_in_memory=args.num_ims_in_memory)

    print('evaluation and feature_saving done')

    # 3: calling diagnostic classifier with input loaded from features
    all_feature_types = ['image_features', 'sender_repr', 'messages', 'receiver_repr']
    if args.analysis_type == "diagnostic":
        if args.feature_type == "all":
            for f in all_feature_types:
                args.feature_type = f
                diagnostic_classify(args)
        else:
            diagnostic_classify(args)
        print(f'Diagnostic done from {f}')

    elif args.analysis_type == "rsa":
        calculate_rsa(args.modelname)

    # 4: deleting existing feature saving except for messages
    to_delete = []
    for set_type in set_list:
        for f in all_feature_types:
            to_delete.append(f'{set_type}_{f}.pickle')
    delete_path = f'saved/{model.model_name}/'
    for f in to_delete:
        if f in os.listdir(delete_path):
            os.remove(delete_path + f)

    print('deleting done')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_type", type=str,
                        help="choose diagnostic or rsa")
    parser.add_argument("--modelname", type=str)
    parser.add_argument("--feature_type", type=str,
                        help="messages, sender_repr, receiver_repr or image_features")
    parser.add_argument("--load_training", action="store_true", default=False)
    parser.add_argument("--load_test", action="store_true", default=False)

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--use_gt", action="store_true", default=False)
    parser.add_argument("--num_ims_in_memory", type=int, default=500)

    parser.add_argument("--message_embedding_dim", type=int, default=128)
    parser.add_argument("--encoder_lstm_hidden_size", type=int, default=128)
    parser.add_argument("--mlp_hidden_dim", type=int, default=64)
    parser.add_argument("--dropout_prob", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pos_weight", type=float, default=1.)



    args = parser.parse_args()

    save_and_diagnose(**vars(args))