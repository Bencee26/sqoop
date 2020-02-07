import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import string
from sqoop.vocab import create_vocab
from sqoop.evaluate import evaluate
from sqoop.utils import get_model_config
from sqoop.save import save_results
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int)
parser.add_argument("--modelname", type=str,
                    help="name of the model in {model_name}/epoch_{epoch_nr}{spec} format")
parser.add_argument("--batch_size", type=str, default=32)
parser.add_argument("--num_samples", type=int, default=1000)
parser.add_argument("--save_messages", type=bool, default=True)
parser.add_argument("--save_features", type=bool, default=True)
parser.add_argument("--calculate_rsa", type=bool, default=True)
parser.add_argument("--num_ims_in_memory", type=int, default=500)

args = parser.parse_args()
s = list(string.ascii_uppercase)

if 'referential' in args.modelname:
    game_type = 'referential'
elif 'multimodal' in args.modelname:
    game_type = 'multimodal'
else:
    ValueError('undefined gametype!')

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

# cleaining files, so wont append to previous
to_delete = ['messages.txt', 'image_features.pickle', 'sender_repr.pickle', 'receiver_repr.pickle', 'test_messages.txt']
delete_path = f'saved/{model.model_name}/'
for f in to_delete:
    if f in os.listdir(delete_path):
        os.remove(delete_path + f)


accuracy, comm_info = evaluate(model, game_type, k, args.batch_size, word2idx, "test", dataset, scale_input,
                               args.save_messages, args.save_features, args.num_ims_in_memory)

save_results(accuracy, args.modelname)

print(f"Accuracy: {accuracy}")