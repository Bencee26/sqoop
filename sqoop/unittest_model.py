import torch
import torch.nn as nn

torch.manual_seed(16)

class VariablesChangeException(Exception):
    pass

def _train_step(model):
    learning_rate = 0.0001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ut_im = torch.rand(64, 3, 64, 64)
    ut_q = torch.randint(0, 40, (64, 3)).type(torch.LongTensor)
    ut_label = torch.randint(0, 2, (64,)).type(torch.LongTensor)

    optimizer.zero_grad()
    pred, _ = model(ut_im, ut_q, False, False)
    loss = criterion(pred, ut_label)
    loss.backward()
    optimizer.step()


def test_variable_change(model):
    model.train()
    params = [np for np in model.named_parameters() if np[1].requires_grad]
    initial_params = [(name, p.clone()) for (name, p) in params]

    _train_step(model)

    for (_, p0), (name, p1) in zip(initial_params, params):
        try:
            if "bottleneck_fc" in name:
                assert not torch.equal(p0, p1)
        except AssertionError:
            raise VariablesChangeException(
                "{var_name} {msg}".format(
                    var_name=name,
                    msg='did not change!')
            )
    print("unittests passed")


def test_loss(model):
    learning_rate = 0.0001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ut_im = torch.rand(64, 3, 64, 64)
    ut_q = torch.randint(0, 40, (64, 3)).type(torch.LongTensor)
    ut_label = torch.randint(0, 2, (64,)).type(torch.LongTensor)

    optimizer.zero_grad()
    pred, _ = model(ut_im, ut_q, False, False)
    loss = criterion(pred, ut_label)
    try:
        assert loss.item() != 0
    except:
        raise ZeroLossException("Loss is zero!")