import h5py
from plot_example_distribution import get_conditions
import os


def filter_relations(original, relation):
    conditions = get_conditions(f'{original}/')
    filter_name = f'{original}_filtered_{relation}'
    os.mkdir(filter_name)
    for set_type in ["training", "validation", "test"]:
        os.mkdir(f'{filter_name}/{set_type}')
        for k in conditions:
            os.mkdir(f'{filter_name}/{set_type}/{k}')

            with h5py.File(f'{original}/{set_type}/{k}/questions.h5', 'r') as hdf:
                questions = hdf.get(f'questions')[:]
            with h5py.File(f'{original}/{set_type}/{k}/samples.h5', 'r') as hdf:
                samples = hdf.get(f'samples')[:]

            filtered_samples = []
            for s in samples:
                if questions[s[1]].decode()[2] == relation[0]:
                    filtered_samples.append(s)

            with h5py.File(f'{filter_name}/{set_type}/{k}/samples.h5', 'w') as hdf:
                hdf.create_dataset('samples', data=filtered_samples)

if __name__ == "__main__":
    original = "data_balanced"
    filter_relations(original, "left_of")
