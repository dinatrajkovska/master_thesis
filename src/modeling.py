from torch import nn


def model_factory(model_type: str, n_feature_types: int):
    if model_type == "regular":
        return piczak_model(n_feature_types)
    elif model_type == "batch_norm":
        return piczak_batchnorm_model(n_feature_types)
    elif model_type == "envnet_v2":
        return envnet_v2()
    else:
        raise ValueError(f"Invalid model type: {model_type}")


# https://www.karolpiczak.com/papers/Piczak2015-ESC-ConvNet.pdf
def piczak_model(n_feature_types):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=n_feature_types,
            out_channels=80,
            kernel_size=(57, 6),
            stride=(1, 1),
        ),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=(4, 3), stride=(1, 3)),
        # nn.Dropout(0.5), Have a look at this dropout again
        nn.Conv2d(in_channels=80, out_channels=80, kernel_size=(1, 3), stride=(1, 1)),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        nn.Flatten(),
        nn.Linear(240, 5000),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Linear(5000, 5000),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Linear(5000, 50),
    )


def piczak_batchnorm_model(n_feature_types: int):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=n_feature_types,
            out_channels=80,
            kernel_size=(57, 6),
            stride=(1, 1),
        ),
        nn.BatchNorm2d(num_features=80),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=(4, 3), stride=(1, 3)),
        nn.Conv2d(in_channels=80, out_channels=80, kernel_size=(1, 3), stride=(1, 1)),
        nn.BatchNorm2d(num_features=80),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        nn.Flatten(),
        nn.Linear(240, 5000),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Linear(5000, 5000),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Linear(5000, 50),
    )


class SwapAxes(nn.Module):
    def __init__(self):
        super(SwapAxes, self).__init__()

    def forward(self, x):
        return x.transpose(2, 1)


def envnet_v2():
    return nn.Sequential(
        # Conv 1 & 2
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 64), stride=(1, 2)),
        nn.BatchNorm2d(num_features=32),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 16), stride=(1, 2)),
        nn.BatchNorm2d(num_features=64),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=(1, 64), stride=(1, 64)),
        SwapAxes(),
        # Conv 3 & 4
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(8, 8), stride=(1, 1)),
        nn.BatchNorm2d(num_features=32),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(8, 8), stride=(1, 1)),
        nn.BatchNorm2d(num_features=32),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=(5, 3), stride=(5, 3)),
        # Conv 5 & 6
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 4), stride=(1, 1)),
        nn.BatchNorm2d(num_features=64),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 4), stride=(1, 1)),
        nn.BatchNorm2d(num_features=64),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        # Conv 7 & 8
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 2), stride=(1, 1)),
        nn.BatchNorm2d(num_features=128),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 2), stride=(1, 1)),
        nn.BatchNorm2d(num_features=128),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        # Conv 9 & 10
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 2), stride=(1, 1)),
        nn.BatchNorm2d(num_features=256),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 2), stride=(1, 1)),
        nn.BatchNorm2d(num_features=256),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        # FC
        nn.Flatten(),
        nn.Linear(256 * 10 * 8, 4096),  # change num of input filters
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 50),
    )
