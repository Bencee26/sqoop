import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
from sqoop.evaluate import evaluate
from sqoop.utils import get_model_config
import torch
import os


parser = argparse.ArgumentParser()
parser.add_argument("--modelname", type=str,
                    help="name of the model in {model_name}/epoch_{epoch_nr}{spec} format")

parser.add_argument("--batch_size", type=str, default=32)
parser.add_argument("--num_ims_in_memory", type=int, default=500)
args = parser.parse_args()


game_type = 'referential'

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

# cleaning files, so wont append to previous
to_delete = ['training_messages.txt']
delete_path = f'saved/{model.model_name}/'
for f in to_delete:
    if f in os.listdir(delete_path):
        os.remove(delete_path + f)


accuracy, comm_info = evaluate(model, game_type, k, args.batch_size, word2idx, "training", dataset, scale_input,
                               save_messages=True, save_features=False, num_ims_in_memory=args.num_ims_in_memory)

