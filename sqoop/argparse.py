import argparse


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--game_type", type=str,
                        help="multimodal or referential")
    parser.add_argument("--k", type=int,
                        help="the #rhs/lhs parameter")
    parser.add_argument("--num_epochs", type=int, default=5000,
                        help="number of training epochs")
    parser.add_argument("--iter_limit", type=int, default=200000,
                        help="maximum number of updates. If this is reached before num_epochs the training is cancelled")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="training batch size")
    parser.add_argument("--arch", default="FiLM",
                        help="model architecture (default: FiLM, alternatively: cnn+lstm)")
    parser.add_argument("--pretrained", type=str,
                        help="continue training an earlier model. Give it in {model_name}/epoch_{epoch_nr}{spec} format")
    parser.add_argument("--use_pretrained_features", type=bool, default=False,
                        help="Use pretrained image features")
    parser.add_argument("--use_ground_truth", type=bool, default=False,
                        help="Switch whether the receiver should use the ground truths instead of the emerging language")
    parser.add_argument("--load_to_memory", type=bool, default=True,
                        help="loading whole training set into memory or loading them batch by batch")
    parser.add_argument("--bottleneck", type=bool, default=False,
                        help="communication bottleneck (default: False)")
    parser.add_argument("--continuous_comm", type=bool, default=False,
                        help="use continuous communication for debugging (default=False)")
    parser.add_argument("--stat_save_interval", type=int, default=500,
                        help="number of iterations between saving statistics (default=500)")
    parser.add_argument("--model_save_interval", type=int, default=10000,
                        help="how many iterations per saving model")
    parser.add_argument("--validation_interval", type=int, default=2000,
                        help="number of iteration between calculating validation score")
    parser.add_argument("--test_model", type=bool, default=False,
                        help="Unit testing the model")
    parser.add_argument("--debug", type=bool, default=False,
                        help="Flag for debugging mode. Does not save stats")
    parser.add_argument("--check_data_loading", type=bool, default=False,
                        help="print question, image, ground truth and label after data preprocessing")
    parser.add_argument("--plot_grad_flow", type=bool, default=False,
                        help="plotting average gradient flow through layers")
    parser.add_argument("--plot_loss", type=bool, default=False)
    parser.add_argument("--cut_image_info", type=bool, default=False,
                        help="zeroing out every info that comes from the image")
    parser.add_argument("--cut_question_info", type=bool, default=False,
                        help="zeroing out every info that comes from the question")
    parser.add_argument("--unbiased_data", type=bool, default=True,
                        help="use the newly created unbiased data")
    parser.add_argument("--dataset", type=str, default="all_samples",
                        help="alternative option: balanced, to use the old dataset with equal number of samples "
                             "per condition, default: all_samples, to use the new dataset")
    parser.add_argument("--num_ims_in_memory", type=int, default=5000,
                        help="number of images to load at once to memory. Used for referential games! "
                             "Be aware that every image contains 4 variations")
    parser.add_argument("--save_messages", type=bool, default=True,
                        help="option for saving messages in evaluation mode")
    parser.add_argument("--train_from_symbolic", action="store_true", default=False,
                        help="training model from ground true values")

    # model hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--scale_input", type=bool, default=True,
                        help="scale input images")
    parser.add_argument("--vocab_size", type=int, default=50,
                        help="bottleneck vocab size")
    parser.add_argument("--max_sentence_length", type=int, default=15,
                        help="maximum sentence lenght")
    parser.add_argument("--num_stem_layers", type=int, default=4,
                        help="number of stem layers (default: 4")
    parser.add_argument("--message_embedding_dim", type=int, default=128,
                        help="dimension of message word embeddings")
    parser.add_argument('--message_lstm_hidden_size', type=int, default=128,
                        help="hidden size of message producing lstm, you can change the botteneck_input_fc's size with "
                             "this param")
    parser.add_argument('--encoder_lstm_hidden_size', type=int, default=128,
                        help="hidden size of message encoding lstm")
    parser.add_argument('--question_embedding_dim', type=int, default=64,
                        help='embedding dim for the question')
    parser.add_argument('--question_rnn_hidden_size', type=int, default=128,
                        help='hidden size of question rnn')
    parser.add_argument('--film_input_size', nargs='+', type=int, default=(64, 4, 4),
                        help='film input size, tuple in shape of (num_channels, height, width)')
    parser.add_argument("--film_channels", type=int, default=64,
                        help="number of film channels (filters)")
    parser.add_argument('--mlp_hidden_dim', type=int, default=256,
                        help="mlp hidden layer dimension")
    parser.add_argument('--dropout_prob', type=float, default=0.0,
                        help="Dropout probability, applied to the fc layers "
                             "(input to message rnn, output after decoder lstm, final mlp) Default: no dropout")

    args = parser.parse_args()

    return args