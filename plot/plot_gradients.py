from matplotlib import pyplot as plt


def plot_grad_flow(model_name, named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    # plt.ylim(0, 0.00005)
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'saved/{model_name}/grad_flow.png')
    plt.close()
