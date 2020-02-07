import torch


def scale_input(images):
    batch_size = images.shape[0]
    channels = images.shape[1]

    max_vals = torch.max(images.view(batch_size, channels, -1),2)[0].view(batch_size, channels, 1, 1)
    min_vals = torch.min(images.view(batch_size, channels, -1),2)[0].view(batch_size, channels, 1, 1)
    midrange = (max_vals + min_vals) / 2
    data_range = max_vals - min_vals
    data_range[data_range==0]=1
    norm_images = (images - midrange)/(data_range/2)

    assert(torch.sum(torch.gt(norm_images, 1)).item() == 0)
    assert(torch.sum(torch.lt(norm_images, -1)).item() == 0)

    return norm_images