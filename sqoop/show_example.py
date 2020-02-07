from PIL import Image
import torch


def show_example(question, image, label, pred, idx2word):
    np_image = (255*image.permute(1, 2, 0).numpy()).astype('uint8')
    pil_image = Image.fromarray(np_image)
    pil_image.show()

    print(f'label: {label.item()}')
    print(f'prediction: {torch.argmax(pred).item()}')

    question_lang = []
    word_idxs = torch.argmax(question, dim=-1)
    for idx in list(question.numpy()):
        question_lang.append(idx2word[idx])
        question_lang.append(' ')
    print(f'question: {question_lang}')