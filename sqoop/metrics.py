import torch
import numpy as np
import scipy.spatial
import scipy.stats


def calc_message_distinctness(message):

    if torch.cuda.is_available():
        message = message.cpu()
    msg = message.data.numpy()
    unique_msgs = np.unique(msg, axis=0)
    message_distinctness = unique_msgs.shape[0]/message.shape[0]

    return message_distinctness


def one_hot(a):
    ncols = a.max() + 1
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def representation_similarity_analysis(
    test_images,
    test_metadata,
    generated_messages,
    hidden_sender,
    hidden_receiver,
    samples=5000,
):

    # one hot encode messages by taking padding into account and transforming to one hot
    # messages = one_hot(generated_messages)
    messages = generated_messages

    assert test_metadata.shape[0] == messages.shape[0]

    sim_image_features = np.zeros(samples)
    sim_metadata = np.zeros(samples)
    sim_messages = np.zeros(samples)
    sim_hidden_sender = np.zeros(samples)
    sim_hidden_receiver = np.zeros(samples)

    for i in range(samples):
        rnd = np.random.choice(len(test_metadata), 2, replace=False)
        s1, s2 = rnd[0], rnd[1]

        sim_metadata[i] = levenshtein(
            test_metadata[s1], test_metadata[s2]
        )

        sim_image_features[i] = scipy.spatial.distance.cosine(
            test_images[s1], test_images[s2]
        )

        # sim_messages[i] = scipy.spatial.distance.cosine(
        #     messages[s1].flatten(), messages[s2].flatten()
        sim_messages[i] = levenshtein(
            messages[s1].flatten(), messages[s2].flatten()
        )

        sim_hidden_sender[i] = scipy.spatial.distance.cosine(
            hidden_sender[s1].flatten(), hidden_sender[s2].flatten()
        )
        sim_hidden_receiver[i] = scipy.spatial.distance.cosine(
            hidden_receiver[s1].flatten(), hidden_receiver[s2].flatten()
        )

    rsa_sr = scipy.stats.pearsonr(sim_hidden_sender, sim_hidden_receiver)[0]
    rsa_si = scipy.stats.pearsonr(sim_hidden_sender, sim_image_features)[0]
    rsa_ri = scipy.stats.pearsonr(sim_hidden_receiver, sim_image_features)[0]
    rsa_mi = scipy.stats.pearsonr(sim_messages, sim_image_features)[0]

    return rsa_sr, rsa_si, rsa_ri, rsa_mi


