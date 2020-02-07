import torch
import torch.nn as nn


class Diagnostic_model(nn.Module):
    def __init__(self,
                 feature_type,
                 feature_length,
                 vocab_size,
                 message_embedding_dim,
                 encoder_lstm_hidden_size,
                 mlp_hidden_dim,
                 num_available_letters,
                 dropout_prob):
        super().__init__()

        self.feature_type = feature_type
        self.feature_length = feature_length
        self.vocab_size = vocab_size
        self.message_embedding_dim = message_embedding_dim
        self.encoder_lstm_hidden_size = encoder_lstm_hidden_size
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_available_letters = num_available_letters
        self.dropout_prob = dropout_prob

        if feature_type == "messages":
            self.message_embedding = nn.Embedding(self.vocab_size, self.message_embedding_dim)
            self.message_encoder_lstm = nn.LSTM(self.message_embedding_dim, self.encoder_lstm_hidden_size, num_layers=1,
                                            batch_first=True)
            self.mlp_input_dim = self.encoder_lstm_hidden_size

        else:
            self.mlp_input_dim = feature_length

        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, self.mlp_hidden_dim),
            nn.Dropout2d(self.dropout_prob),
            nn.ReLU(),
            # nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            # nn.Dropout2d(self.dropout_prob),
            # nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.num_available_letters)
        )

    def forward(self, feature):
        if self.feature_type == "messages":
            embedded_message = self.message_embedding(feature)
            encoded_message = self.message_encoder_lstm(embedded_message)[1][0]
            encoded_message = encoded_message.view(-1, self.encoder_lstm_hidden_size)
            mlp_input_features = encoded_message
        else:
            mlp_input_features = feature
        out = self.mlp(mlp_input_features)

        return out


