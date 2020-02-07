import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from sqoop.calc_stem_output_shape import calc_conv_output_shape
from sqoop.metrics import calc_message_distinctness
from sqoop.utils import batch_2_onehot
from sqoop.save import save_message, save_ground_truths
import sys


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=imm_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(in_channels=imm_channels, out_channels=out_features, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        out = x * self.weight(y).unsqueeze(2).unsqueeze(3) + self.bias(y).unsqueeze(2).unsqueeze(3)
        out = self.bn2(out)
        out = F.relu(out)
        return out


class Sqoop_model(nn.Module):

    def __init__(self,
                 model_name,
                 idx2word,
                 game_type="referential",
                 image_size=(64, 64),
                 batch_size=64,
                 use_pretrained_features=False,
                 use_ground_truth=False,
                 train_from_symbolic=False,
                 num_chars_per_image=2,
                 arch="FiLM",
                 num_stem_filters=128,
                 num_stem_layers=4,
                 conv_kernel_size=3,
                 conv_stride=1,
                 conv_padding=1,
                 bottleneck=False,
                 continuous_communication=False,
                 max_sentence_length=10,
                 vocab_size=26,
                 message_embedding_dim=128,
                 embedding_init_std = 0.001,
                 message_lstm_hidden_size=128,
                 tau=1.2,
                 greedy=True,
                 encoder_lstm_hidden_size=128,
                 film_input_size=(64, 4, 4),
                 film_channels=64,
                 num_words_in_question_vocab=40,
                 question_embedding_dim=64,
                 question_rnn_hidden_size=128,
                 question_rnn_num_layers=1,
                 dropout_prob=0,
                 mlp_hidden_dim=256,
                 ):
        super().__init__()

        self.model_name = model_name
        self.idx2word = idx2word
        self.game_type = game_type
        self.image_size = image_size
        self.use_pretrained_features = use_pretrained_features
        self.use_ground_truth = use_ground_truth
        self.train_from_symbolic = train_from_symbolic
        self.arch = arch
        self.batch_size = batch_size
        self.num_stem_filters = num_stem_filters
        self.num_stem_layers = num_stem_layers
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.bottleneck = bottleneck
        self.num_chars_per_image = num_chars_per_image
        self.continuous_communication = continuous_communication
        self.max_sentence_length = max_sentence_length
        self.vocab_size = vocab_size
        self.message_embedding_dim = message_embedding_dim
        self.embedding_init_std = embedding_init_std
        self.message_lstm_hidden_size = message_lstm_hidden_size
        self.tau = tau
        self.greedy = greedy
        self.encoder_lstm_hidden_size = encoder_lstm_hidden_size
        self.question_embedding_dim = question_embedding_dim
        self.question_rnn_hidden_size = question_rnn_hidden_size
        self.num_words_in_question_vocab = num_words_in_question_vocab
        self.question_rnn_num_layers = question_rnn_num_layers
        self.film_input_size = film_input_size
        self.film_channels = film_channels
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout_prob = dropout_prob

        self.use_gpu = torch.cuda.is_available()

        self.drop = nn.Dropout2d(self.dropout_prob)

        self.conv_channel_sizes = [3] + [num_stem_filters] * self.num_stem_layers

        def conv_block(in_f, out_f, *args, **kwargs):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, *args, **kwargs),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            # building conv layers
        conv_blocks = [conv_block(in_f, out_f, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
                   for in_f, out_f in zip(self.conv_channel_sizes, self.conv_channel_sizes[1:])]

        if not self.train_from_symbolic:
            self.stem_conv = nn.Sequential(*conv_blocks)

            # calculating the shapes
            self.stem_out_size = calc_conv_output_shape(num_stem_filters, num_stem_layers, image_size, conv_kernel_size,
                                                        conv_stride, conv_padding)

            self.n_stem_features = int(np.prod(self.stem_out_size))

        else:
            self.n_stem_features = self.message_lstm_hidden_size

        # bottleneck
        if self.bottleneck:
            if not self.train_from_symbolic:
                self.bottleneck_in_fc = nn.Linear(self.n_stem_features, self.message_lstm_hidden_size)
            else:
                self.bottleneck_in_fc = nn.Linear(2*num_chars_per_image, self.message_lstm_hidden_size)

            self.message_embedding = nn.Parameter(torch.randn((vocab_size, message_embedding_dim), dtype=torch.float32) * self.embedding_init_std)

            self.lstm_cell = nn.LSTMCell(self.message_embedding_dim, self.message_lstm_hidden_size)
            self.hidden2vocab = nn.Linear(self.message_lstm_hidden_size, self.vocab_size)

            self.bound_token_idx = 0

            self.message_encoder_lstm = nn.LSTM(self.message_embedding_dim, self.encoder_lstm_hidden_size, num_layers=1,
                                                batch_first=True)
            # self.FiLM_num_feats = np.prod(np.array(list(self.film_input_size)))

            self.aff_transform = nn.Linear(self.encoder_lstm_hidden_size, self.n_stem_features)

        # Question embedding
        # self.question_embedding = nn.Embedding(self.num_words_in_question_vocab, question_embedding_dim)
        # self.question_rnn = nn.LSTM(self.question_embedding_dim, self.question_rnn_hidden_size, self.question_rnn_num_layers, batch_first=True)

        # Fusing information

        # if not bottleneck:
        #     self.film_input_size = self.stem_out_size
        #
        # if self.arch == "FiLM":
        #     num_module = 2
        #     self.controllers = []
        #     for ni in range(num_module):
        #         if ni < num_module - 1:
        #             mod = FiLM(
        #                 in_features=self.question_rnn_hidden_size,
        #                 out_features=self.film_channels,
        #                 in_channels=self.film_input_size[0],
        #                 imm_channels=self.film_channels)
        #         else:
        #             mod = FiLM(
        #                 in_features=self.question_rnn_hidden_size,
        #                 out_features=self.film_channels,
        #                 in_channels=self.film_channels,
        #                 imm_channels=self.film_channels)
        #         self.controllers.append(mod)
        #         self.add_module('FiLM_' + str(ni), mod)
        #     self.film_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        #
        #     self.film_out_size = (film_channels, self.film_input_size[1] // 2, self.film_input_size[2] // 2)
        #     self.fc_input_size = int(np.prod(self.film_out_size))
        #
        # elif self.arch == "cnn+lstm":
        #     self.fc_input_size = self.n_stem_features + self.question_rnn_hidden_size
        #
        # else:
        #     raise ValueError("Invalid model architecture. Choose cnn+lstm or FiLM")
        #
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.fc_input_size, self.mlp_hidden_dim),
        #     nn.Dropout2d(self.dropout_prob),
        #     nn.Linear(self.mlp_hidden_dim, 2)
        # )

    def forward(self, image, question=None, feature_saving=False, cut_image_info=False, cut_question_info=False,
                ground_truth=None, save_messages=False, save_features=False):

        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            device = 'cuda'
        else:
            device = 'cpu'
        assert not (feature_saving and self.training)
        self.batch_size = image.shape[0]
        im_features = []

        if not self.train_from_symbolic:
            for i in range(image.shape[1]):
                im_features.append(self.stem_conv(image[:, i, :, :, :]).view(self.batch_size, -1, 1))
            image_info = im_features[0]
            candidate_im_features = im_features[1:]

        else:
            candidate_im_features = []
            ground_truth = ground_truth.type(torch.FloatTensor).to(device)
            image_info = ground_truth[:, 0, :]
            for i in range(image.shape[1]-1):
                candidate_im_features.append(self.bottleneck_in_fc(ground_truth[:, i+1, :]).view(self.batch_size, -1, 1))



        #detaching gradients
        candidate_im_features = [candidate.detach() for candidate in candidate_im_features]

        if self.bottleneck:

            if self.training:
                message = [torch.zeros((self.batch_size, self.vocab_size), dtype=torch.float32)]
                if self.use_gpu:
                    message[0] = message[0].cuda()
                message[0][:, self.bound_token_idx] = 1.0
            else:
                message = [torch.full((self.batch_size,), fill_value=self.bound_token_idx, dtype=torch.int64)]
                if self.use_gpu:
                    message[0] = message[0].cuda()

            # h0, c0
            flattened_im_features = image_info.view(self.batch_size, -1)
            sender_representation = self.drop(self.bottleneck_in_fc(flattened_im_features))
            h = sender_representation
            c = torch.zeros((self.batch_size, self.message_lstm_hidden_size))

            if self.use_gpu:
                c = c.cuda()

            entropy = 0.0

            # produce words one by one
            for i in range(self.max_sentence_length):

                emb = torch.matmul(message[-1], self.message_embedding) if self.training else self.message_embedding[message[-1]]

                h, c = self.lstm_cell(emb, (h, c))

                vocab_scores = self.drop(self.hidden2vocab(h))
                p = F.softmax(vocab_scores, dim=1)
                entropy += Categorical(p).entropy()

                if self.training:

                    rohc = RelaxedOneHotCategorical(self.tau, p)
                    token = rohc.rsample()

                    # Straight-through part
                    if not self.continuous_communication:
                        token_hard = torch.zeros_like(token)
                        token_hard.scatter_(-1, torch.argmax(token, dim=-1, keepdim=True), 1.0)
                        token = (token_hard - token).detach() + token

                else:
                    if self.greedy:
                        _, token = torch.max(p, -1)
                    else:
                        token = Categorical(p).sample()

                message.append(token)
            message = torch.stack(message, dim=1)

            if self.training:
                _, m = torch.max(message, dim=-1)
            else:
                m = message

            md = calc_message_distinctness(m)


            # If we feed the ground_truth to the receiver, we simply hijack the message here
            if self.use_ground_truth:
                if self.training:
                    message = batch_2_onehot(ground_truth, self.max_sentence_length, self.vocab_size)
                else:
                    message = ground_truth.type(torch.LongTensor)
                if self.use_gpu:
                    message = message.cuda()


            # Receiver part

            h = torch.zeros((self.batch_size, self.encoder_lstm_hidden_size))
            c = torch.zeros((self.batch_size, self.encoder_lstm_hidden_size))

            if self.use_gpu:
                h = h.cuda()
                c = c.cuda()

            emb = torch.matmul(message, self.message_embedding) if self.training else self.message_embedding[message]
            _, (h, c) = self.message_encoder_lstm(emb, (h[None, ...], c[None, ...]))
            hidden_receiver = h[0]
            bottleneck_out = self.drop(self.aff_transform(hidden_receiver))

            image_info = bottleneck_out

            #todo recomment
            # comm_info = {'entropy': torch.mean(entropy).item() / (self.max_sentence_length+ 1e-7), 'md': md}
            comm_info = {'entropy': 0, 'md': 0}
            if save_features:
                comm_info['image_features'] = flattened_im_features
                comm_info['sender_repr'] = sender_representation
                comm_info['receiver_repr'] = hidden_receiver
            if save_messages:
                comm_info['message'] = m

        # in case no communication bottleneck
        else:
            comm_info = None

        orig_im_features = image_info.view(self.batch_size, 1, -1)
        out = torch.zeros(self.batch_size, len(candidate_im_features)).type(torch.FloatTensor)
        if self.use_gpu:
            out = out.cuda()
        for i in range(len(candidate_im_features)):
            out[:, i] = torch.bmm(orig_im_features, candidate_im_features[i]).squeeze()

        return out, comm_info
