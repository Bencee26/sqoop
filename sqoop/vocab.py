import string


def create_vocab(S):
    words = S + ['left_of', 'right_of', "above", "below"]

    idx2word = {i: words[i] for i in range(len(words))}
    word2idx = {words[i]: i for i in range(len(words))}

    return idx2word, word2idx
