import torch
import argparse
import numpy as np
from sqoop.dataloader import load_data, load_diagnostic_batch
from sqoop.utils import get_model_config
import torch.optim as optim
from sqoop.diagnostic_model import Diagnostic_model
from calculate_metrics import get_messages
import pickle
from sqoop.save import save_diagnostic


def get_metrics(decision, batch_label):
    metrics = {}

    eps = 1e-7
    accuracy = torch.mean((decision == batch_label).type(torch.FloatTensor)).item()

    all_pos = torch.sum(decision).item()
    all_neg = decision.numel() - all_pos

    tp = torch.sum((batch_label == 1) * (decision == 1)).item()
    tpr = tp / (all_pos + eps)
    tn = torch.sum((batch_label == 0) * (decision == 0)).item()
    tnr = tn / (all_neg + eps)

    balanced_accuracy = (tpr + tnr) / 2

    precision = tp / (all_pos+eps)

    fn = torch.sum((batch_label == 1) * (decision == 0)).item()
    recall = tp / (tp+fn)

    f1 = 2 * precision * recall / (precision + recall + eps)

    pos_per_sample = all_pos / decision.shape[0]

    metrics['accuracy'] = accuracy
    metrics['balanced_accuracy'] = balanced_accuracy
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    metrics['pos_per_sample'] = pos_per_sample

    return metrics


def print_results(epoch, set_type, acc, bacc, prec, rec, f1, pps, rec_at_k, rec_at_2k):
    if set_type == 'validation' or set_type == 'test':
        print('')
    print(f'Epoch {epoch}, On {set_type} set '
          f'accuracy: {acc}, '
          f'balanced accuracy: {bacc},'
          f'\n precision: {prec}, '
          f'recall: {rec}, '
          f'F1 score: {f1}'
          f', number of positive guesses per example: {pps}\n'
          f'Recall@k: {rec_at_k} '
          f'Recall@2k: {rec_at_2k}'
          )
    if set_type == 'validation' or set_type == 'test':
        print('')


def eval_model(model, messages, ground_truths, num_available_letters, device, use_gt, batch_size, num_chars_on_image):
    eval_size = len(messages)
    c = 0
    accuracy = []
    balanced_accuracy = []
    precision = []
    recall = []
    f1 = []
    pos_per_sample = []
    rec_at_k = []
    rec_at_2k = []

    while c < eval_size:
        batch_message, batch_label = load_diagnostic_batch(c, batch_size, eval_size, messages,
                                                           ground_truths, num_available_letters, use_gt=use_gt)

        batch_label = batch_label.type(torch.ByteTensor)
        if device == 'cuda':
            batch_label = batch_label.cuda()
        model.eval()

        out = model(batch_message)
        decision = out > 0

        metrics = get_metrics(decision, batch_label)
        accuracy.append(metrics['accuracy'])
        balanced_accuracy.append(metrics['balanced_accuracy'])
        precision.append(metrics['precision'])
        recall.append(metrics['recall'])
        f1.append(metrics['f1'])
        pos_per_sample.append(metrics['pos_per_sample'])

        rec_at_k.append(get_rec_at_k(out, batch_label, num_chars_on_image, num_chars_on_image))
        rec_at_2k.append(get_rec_at_k(out, batch_label, 2 * num_chars_on_image, num_chars_on_image))

        c += batch_size

    return np.round(np.mean(np.array(accuracy)), 3), np.round(np.mean(np.array(balanced_accuracy)), 3), \
           np.round(np.mean(np.array(precision)), 3), np.round(np.mean(np.array(recall)), 3), \
           np.round(np.mean(np.array(f1)), 3), np.round(np.mean(np.array(pos_per_sample)), 2), \
           np.round(np.mean(rec_at_k), 3), np.round(np.mean(rec_at_2k), 3)


def load_feature(feature_type, model_name, set_list):

    feats = []
    for set_type in set_list:

        path = f'saved/{model_name}/{set_type}_{feature_type}.pkl'
        feature = []
        with open(path, 'rb') as file:
            while True:
                try:
                    feature.append(pickle.load(file))
                except EOFError:
                    break
        feature = torch.stack(feature)
        feats.append(feature)

    if len(feats) > 1:
        feats = torch.cat(tuple([f for f in feats]), 0)
    else:
        feats = feats[0]
    return feats


def loadsets_to_list(load_training, load_test):

    assert load_training or load_test
    set_list = []
    if load_training:
        set_list.append('training')
    if load_test:
        set_list.append('test')
    return set_list


def get_rec_at_k(out, labels, k, num_chars_on_image):
    first_k = torch.topk(out, k)[1]
    num_correct = 0
    batch_size = out.shape[0]
    for i in range(batch_size):
        for j in range(k):
            if labels[i][first_k[i, j]] == 1:
                num_correct += 1
    rec_at_k = num_correct/(batch_size*num_chars_on_image)

    return rec_at_k


def diagnostic_classify(args):

    set_list = loadsets_to_list(args.load_training, args.load_test)

    config_dict = get_model_config(args.modelname)
    game_type = config_dict['game_type']
    dataset = config_dict['dataset']

    num_chars_on_image = int(dataset[-1])
    print(f'num_chars_on_image: {num_chars_on_image}')
    print(f'Training diagnostic classifier from: {args.feature_type} \n\n\n')

    scale_input = config_dict['scale_input']
    word2idx = config_dict['word2idx']
    num_available_letters = len(word2idx)

    all_ground_truths = []
    if args.load_test:
         _, all_ground_truths_test, _, _, _ = load_data(game_type, "test", word2idx, scale_input, dataset, discard_image=True)
         all_ground_truths.append(all_ground_truths_test)

    if args.load_training:
        _, all_ground_truths_training, _, _, _ = load_data(game_type, "training", word2idx, scale_input, dataset, discard_image=True)
        all_ground_truths.append(all_ground_truths_training)

    if len(all_ground_truths) > 1:
        all_ground_truths = torch.cat([gt for gt in all_ground_truths], 0)
    else:
        all_ground_truths = all_ground_truths[0]

    if args.feature_type == "messages":
        all_messages = []
        if args.load_test:
            all_messages_test = get_messages(args.modelname, "test")
            all_messages.append(all_messages_test)
        if args.load_training:
            all_messages_training = get_messages(args.modelname, "training")
            all_messages.append(all_messages_training)

        if len(all_messages) > 1:
            all_messages = torch.cat([m for m in all_messages], 0)
        else:
            all_messages = all_messages[0]
        assert (len(all_ground_truths) == len(all_messages))
        all_features = all_messages

    else:
        all_features = load_feature(args.feature_type, args.modelname, set_list)


    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if device == 'cuda':
        all_features = all_features.cuda()
        all_ground_truths = all_ground_truths.cuda()

    num_samples = len(all_ground_truths)

    idx = torch.randperm(num_samples)
    all_features = all_features[idx]
    all_ground_truths = all_ground_truths[idx]

    train_split = 0.9
    valid_split = 0.05

    train_end_idx = int(num_samples * train_split)
    valid_end_idx = train_end_idx + int(num_samples * valid_split)

    training_features = all_features[:train_end_idx]
    validation_features = all_features[train_end_idx : valid_end_idx]
    test_features = all_features[valid_end_idx:]

    training_ground_truths = all_ground_truths[:train_end_idx]
    validation_ground_truths = all_ground_truths[train_end_idx:valid_end_idx]
    test_ground_truths = all_ground_truths[valid_end_idx:]

    train_size = len(training_features)

    vocab_size = torch.max(all_features).item()+1

    feature_length = len(all_features[0])

    model = Diagnostic_model(args.feature_type,
                             feature_length,
                             vocab_size,
                             args.message_embedding_dim,
                             args.encoder_lstm_hidden_size,
                             args.mlp_hidden_dim,
                             num_available_letters,
                             args.dropout_prob)

    if device == 'cuda':
        model.cuda()

    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([args.pos_weight] * num_available_letters)).to(device)

    for epoch in range(args.num_epochs):

        c = 0
        accuracy = []
        balanced_accuracy = []
        precision = []
        recall = []
        f1 = []
        pos_per_sample = []
        rec_at_k = []
        rec_at_2k = []

        while c < train_size:
            batch_message, batch_label = load_diagnostic_batch(c, args.batch_size, train_size, training_features,
                                                               training_ground_truths, num_available_letters,
                                                               use_gt=args.use_gt)
            batch_message.detach_()

            batch_label = batch_label.type(torch.ByteTensor)
            if device == 'cuda':
                batch_label = batch_label.cuda()

            model.train()
            optimizer.zero_grad()

            out = model(batch_message)

            decision = out > 0

            metrics = get_metrics(decision, batch_label)
            accuracy.append(metrics['accuracy'])
            balanced_accuracy.append(metrics['balanced_accuracy'])
            precision.append(metrics['precision'])
            recall.append(metrics['recall'])
            f1.append(metrics['f1'])
            pos_per_sample.append(metrics['pos_per_sample'])

            rec_at_k.append(get_rec_at_k(out, batch_label, num_chars_on_image, num_chars_on_image))
            rec_at_2k.append(get_rec_at_k(out, batch_label, 2*num_chars_on_image, num_chars_on_image))

            batch_label = batch_label.type(torch.FloatTensor)
            if device == 'cuda':
                batch_label = batch_label.cuda()

            loss = criterion(out, batch_label)
            loss.backward()
            optimizer.step()

            c += args.batch_size

        print_results(epoch, 'training', np.round(np.mean(accuracy),2), np.round(np.mean(balanced_accuracy),2),
                      np.round(np.mean(precision),2), np.round(np.mean(recall), 2), np.round(np.mean(f1), 2),
                      np.round(np.mean(pos_per_sample), 2), np.round(np.mean(rec_at_k), 2), np.round(np.mean(rec_at_2k), 2))

        if epoch % args.eval_interval == 0:
            accuracy, balanced_accuracy, precision, recall, f1, pos_per_sample, r_at_k, r_at_2k = eval_model(model, validation_features, validation_ground_truths, num_available_letters, device,
                                                                                            args.use_gt, args.batch_size, num_chars_on_image)

            stats = {'accuracy': accuracy, 'balanced_accuracy': balanced_accuracy, 'precision': precision, 'recall': recall,
                     'f1': f1, 'pos_per_sample': pos_per_sample, 'recall_at_k': r_at_k, 'recall_at_2k': r_at_2k}
            save_diagnostic(args.modelname, args.feature_type, stats)
            print_results(epoch, "validation", accuracy, balanced_accuracy, precision, recall, f1, pos_per_sample, r_at_k, r_at_2k)

    accuracy, balanced_accuracy, precision, recall, f1, pos_per_sample, r_at_k, r_at_2k = eval_model(model, test_features, test_ground_truths, num_available_letters, device,
                                                                                                     args.use_gt, args.batch_size, num_chars_on_image)
    print_results(epoch, "test", accuracy, balanced_accuracy, precision, recall, f1, pos_per_sample, r_at_k, r_at_2k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str)
    parser.add_argument("--feature_type", type=str)
    parser.add_argument("--load_training", type=bool, default=True)
    parser.add_argument("--load_test", type=bool, default=True)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--use_gt", type=bool, default=False)

    parser.add_argument("--message_embedding_dim", type=int, default=128)
    parser.add_argument("--encoder_lstm_hidden_size", type=int, default=128)
    parser.add_argument("--mlp_hidden_dim", type=int, default=64)
    parser.add_argument("--dropout_prob", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pos_weight", type=float, default=6.)

    args = parser.parse_args()

    diagnostic_classify(args)