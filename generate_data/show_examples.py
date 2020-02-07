import h5py
import argparse
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os


def show_examples(path):

    data_name = get_name(path)

    text_color = (200, 200, 0)
    font_size = 14
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerif.ttf", font_size, encoding="unic")
    except:
        font = ImageFont.load_default()

    if 'referential' in path:
        game_type = "referential"
    else:
        game_type = "multimodal"

    with h5py.File(f'{path}/training/images.h5', 'r') as hdf:
        images = hdf.get('images')[:]
        ground_truths = hdf.get('ground_truths')[:]
        if game_type == "referential":
            labels = hdf.get('labels')[:]

    if game_type == "referential":
        n_examples = 4
        idxs = sorted(list(np.random.choice(range(len(images)), n_examples, replace=False)))
        ims = images[idxs]
        gts = ground_truths[idxs]
        labels = labels[idxs]

        num_ims_per_sample = len(ims[0])
        num_distractors = num_ims_per_sample - 2
        im = Image.fromarray(ims[0, 0])
        x, y = im.size

        separator_space = 50
        text_space_x = 80
        text_space_y = 60
        space = 30
        x += space
        y += space
        fig = Image.new('RGB', (text_space_x + separator_space + x * num_ims_per_sample, text_space_y + y * n_examples))
        push_right = {}
        for i in range(len(ims)):
            for j in range(num_ims_per_sample):
                if j == 0:
                    push_right[j] = 0
                else:
                    push_right[j] = separator_space
                px, py = j*x + text_space_x + push_right[j], i*y + text_space_y
                fig.paste(Image.fromarray(ims[i, j]), (px, py))
                # drawing box around positive example
                if j == labels[i]+1:
                    draw = ImageDraw.Draw(fig)
                    draw.rectangle([(px, py), (px+x-space, py+y-space)], outline=(255, 255, 255))

        data_types = ['original', '', 'candidates', '']

        draw = ImageDraw.Draw(fig)
        for i in range(num_ims_per_sample):
            draw.text((text_space_x + i*x + push_right[i], 5), data_types[i], text_color, font=font)
        for i in range(n_examples):
            draw.text((5, text_space_y + (x-space)/2 + i*x), f'example {i+1}', text_color, font=font)

    if game_type is "multimodal":

        with h5py.File(f'{path}/training/questions.h5', 'r') as hdf:
            questions = hdf.get('questions')[:]

        with h5py.File(f'{path}/training/samples.h5', 'r') as hdf:
            samples = hdf.get('samples')[:]

        n_examples = 4
        idxs = sorted(list(np.random.choice(range(len(samples)), n_examples, replace=False)))
        samples = samples[idxs]

        ims = []
        gts = []
        qs = []
        labels = []

        for i in range(n_examples):
            im_idx = samples[i][0]
            ims.append(images[im_idx])
            gt = ground_truths[im_idx]
            gt1, gt2 = gt[0], gt[1]
            gt1_decoded = [c.decode() for c in gt1]
            gt2_decoded = [c.decode() for c in gt2]
            gt_decoded = gt1_decoded + gt2_decoded
            gt_string = "[" + "".join(gt1_decoded) + "],[" + "".join(gt2_decoded) + "]"
            gts.append(gt_string)

            q_idx = samples[i][1]
            qs.append(questions[q_idx].decode())

            labels.append(str(samples[i][2]))

        im = Image.fromarray(ims[0])
        x, y = im.size

        text_space_x = 80
        text_space_y = 30
        ground_truth_space = 150
        question_space = 90
        label_space = 40
        space = 30
        x += space
        y += space
        fig = Image.new('RGB', (text_space_x + x + ground_truth_space + question_space + label_space, text_space_y + y * n_examples))

        for i in range(n_examples):
            px, py = text_space_x, i * y + text_space_y
            fig.paste(Image.fromarray(ims[i]), (px, py))

        draw = ImageDraw.Draw(fig)
        for i in range(n_examples):
            if i == 0:
                draw.text((text_space_x, 5), 'image', text_color, font=font)
                draw.text((text_space_x + x, 5), 'ground truth', text_color, font=font)
                draw.text((text_space_x + x + ground_truth_space , 5), 'question', text_color, font=font)
                draw.text((text_space_x + x + ground_truth_space + question_space, 5), 'label', text_color, font=font)

            draw.text((5, text_space_y + (x - space) / 2 + i * x), f'example {i + 1}', text_color, font=font)
            draw.text((text_space_x + x, y*i + text_space_y + (x - space)/2), gts[i], text_color, font=font)
            draw.text((text_space_x + x + ground_truth_space, y*i + text_space_y + (x - space)/2), qs[i], text_color, font=font)
            draw.text((text_space_x + x + ground_truth_space + question_space + 10, y*i + text_space_y + (x - space)/2), labels[i], text_color, font=font)

            
    if 'figures' not in os.listdir():
        os.mkdir('figures')
    fig.show()
    fig.save(f'{path}/example', format='png')


def get_name(path):
    if '/' in path:
        if 'multimodal' in path:
            idx = path.find('multimodal')
            name = path[idx+len('multimodal')+1:]
        elif 'referential' in path:
            idx = path.find('referential')
            name = path[idx+len('referential')+1:]
    if '/' in name:
        idx = name.find('/')
        name = name[:idx]
    else:
        name = path

    return name


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()

    show_examples(args.path)