from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn

def plot_parameter_chart(model_name, model):

    main_layers = ["stem", "bottleneck", "question", "film", "mlp"]

    total_params = sum([p[1].numel() for p in model.named_parameters()])
    stem_params = sum([p.numel() for p in model.stem_conv.parameters()])

    bottleneck_params = 0
    if model.bottleneck:
        bottleneck_in_fc_params = sum([p.numel() for p in model.bottleneck_in_fc.parameters()])
        bottleneck_params += bottleneck_in_fc_params
        lstm_cell_params = sum([p.numel() for p in model.lstm_cell.parameters()])
        bottleneck_params += lstm_cell_params
        hidden2vocab_params = sum([p.numel() for p in model.hidden2vocab.parameters()])
        bottleneck_params += hidden2vocab_params
        message_embedding_params = model.message_embedding.shape[0] * model.message_embedding.shape[1]
        bottleneck_params += message_embedding_params
        message_encoder_lstm_params = sum([p.numel() for p in model.message_encoder_lstm.parameters()])
        bottleneck_params += message_encoder_lstm_params
        aff_transform_params = sum([p.numel() for p in model.aff_transform.parameters()])
        bottleneck_params += aff_transform_params


    question_params = 0
    question_embedding_params = sum([p.numel() for p in model.question_embedding.parameters()])
    question_params += question_embedding_params
    question_rnn_params = sum([p.numel() for p in model.question_rnn.parameters()])
    question_params += question_rnn_params

    film0_params = sum([p.numel() for p in model.FiLM_0.parameters()])
    film1_params = sum([p.numel() for p in model.FiLM_1.parameters()])
    film_params = film0_params + film1_params

    mlp_layer_params = []
    for i in range(len(model.mlp)):
        if isinstance(model.mlp[i], nn.Linear):
            mlp_layer_params.append(sum([p.numel() for p in model.mlp[i].parameters()]))
    mlp_params = sum(mlp_layer_params)

    param_dict = {"stem": stem_params,
                  "bottleneck": bottleneck_params,
                  "question": question_params,
                  "film": film_params,
                  "mlp": mlp_params}

    num_sub_layers = {"stem": 1,
                      "bottleneck": 7,
                      "question": 2,
                      "film": 2,
                      "mlp": len(mlp_layer_params)}

    outer_circle_dict = {'stem': stem_params}

    if model.bottleneck:
        outer_circle_dict['bottleneck_in_fc'] = bottleneck_in_fc_params
        outer_circle_dict['lstm_cell'] = lstm_cell_params
        outer_circle_dict['hidden2vocab'] = hidden2vocab_params
        outer_circle_dict['message_embedding'] = message_embedding_params
        outer_circle_dict['messae_encoder_lstm'] = message_encoder_lstm_params
        outer_circle_dict['aff_transform'] = aff_transform_params

    outer_circle_dict['question_embedding'] = question_embedding_params
    outer_circle_dict['question_rnn_params'] = question_rnn_params

    outer_circle_dict['film1'] = film0_params
    outer_circle_dict['film2'] = film1_params

    for i in range(len(mlp_layer_params)):
        layer_name = f'mlp_{i+1}'
        outer_circle_dict[layer_name] = mlp_layer_params[i]

    # remove stem if its not training
    if model.use_pretrained_features:
        main_layers.remove("stem")
        del param_dict["stem"]
        del outer_circle_dict["stem"]
        del [num_sub_layers["stem"]]
        total_params -= stem_params

    param_names = []
    num_params = []
    for k, v in param_dict.items():
        param_names.append(k)
        num_params.append(v)

    outer_param_names = []
    outer_num_params = []
    for k, v in outer_circle_dict.items():
        outer_param_names.append(k)
        outer_num_params.append(v)

    cmap = plt.get_cmap("tab20b")
    num_colors = len(param_names) + len(outer_param_names)
    inner_c_vec = []
    outer_c_vec = []
    num_same = 0
    layer_idx = 0
    for i in range(num_colors):
        if num_same <= num_sub_layers[main_layers[layer_idx]]:
            if num_same == 0:
                inner_c_vec.append(i)
            else:
                outer_c_vec.append(i)
            num_same += 1
        else:
            num_same = 1
            inner_c_vec.append(i)
            layer_idx += 1

    plt.title(f'Total number of trainable parameters: {total_params}')

    outer_colors = cmap(np.arange(len(outer_c_vec)))

    wedges_out, _ = plt.pie(outer_num_params,
                            wedgeprops=dict(width=0.6),
                            labels=outer_num_params,
                            counterclock=False, startangle=90,
                            colors=outer_colors)

    plt.legend(wedges_out, outer_param_names, title="parameters", loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'saved/{model_name}/parameters.png')
    plt.close()
