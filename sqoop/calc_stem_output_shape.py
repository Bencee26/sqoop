def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1

    return h, w


def calc_conv_output_shape(num_filters, num_layers, h_w, kernel_size=1, stride=1, pad=0, dilatation=1):
    for _ in range(num_layers):
        h_w = conv_output_shape(h_w, kernel_size, stride, pad, dilatation)
    return num_filters, h_w[0]//(2 ** num_layers), h_w[1]//(2 ** num_layers)
