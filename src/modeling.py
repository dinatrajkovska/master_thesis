from torch import nn


def model_factory(model_type: str, n_feature_types: int):
    if model_type == "regular":
        return piczak_model(n_feature_types)
    elif model_type == "batch_norm":
        return piczak_batchnorm_model(n_feature_types)
    elif model_type == "dcnn":
        return dcnn_model(n_feature_types)
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
        nn.Dropout(0.5),  # Have a look at this dropout again
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


def dcnn_model(n_feature_types):
    # Environment Sound Classification using Multiple Feature Channels and Deep Convolutional Neural Networks
    return nn.Sequential(
        # Block 1
        nn.Conv2d(
            in_channels=n_feature_types,
            out_channels=32,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        ),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        ),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=(1, 1), stride=1, padding=0
        ),
        nn.LeakyReLU(),
        nn.BatchNorm2d(num_features=32),
        nn.MaxPool2d(kernel_size=(1, 2), padding=0),
        # Block 2
        nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0),
        ),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0),
        ),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=(1, 1), stride=1, padding=0
        ),
        nn.LeakyReLU(),
        nn.BatchNorm2d(num_features=32),
        nn.MaxPool2d(kernel_size=(4, 1), padding=(1, 0)),
        # Block 3
        nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        ),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        ),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(1, 1), stride=1, padding=0
        ),
        nn.LeakyReLU(),
        nn.BatchNorm2d(num_features=64),
        nn.MaxPool2d(kernel_size=(1, 2), padding=0),
        # Block 4
        nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0),
        ),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0),
        ),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(1, 1), stride=1, padding=0
        ),
        nn.LeakyReLU(),
        nn.BatchNorm2d(num_features=64),
        nn.MaxPool2d(kernel_size=(4, 1), padding=0),
        # Block 5
        nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(5, 3),
            stride=1,
            padding=(2, 1),
        ),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(5, 3),
            stride=1,
            padding=(2, 1),
        ),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1, padding=0
        ),
        nn.LeakyReLU(),
        nn.BatchNorm2d(num_features=128),
        nn.MaxPool2d(kernel_size=(4, 2), padding=(1, 0)),
        # Block 6
        nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(5, 3),
            stride=1,
            padding=(2, 1),
        ),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(5, 3),
            stride=1,
            padding=(2, 1),
        ),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(1, 1), stride=1, padding=0
        ),
        nn.LeakyReLU(),
        nn.BatchNorm2d(num_features=256),
        nn.MaxPool2d(kernel_size=(4, 2), padding=(1, 0)),
        nn.Flatten(),
        nn.Linear(512, 512),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 50),
    )


# Class-based model
class MainBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding, pool_size, maxpool_pad
    ):
        super(MainBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
        )
        self.leaky_relu = nn.LeakyReLU()
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=pool_size, padding=maxpool_pad)

    def forward(self, x):
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)
        out = self.conv3(out)
        out = self.leaky_relu(out)
        out = self.batchnorm(out)
        out = self.maxpool(out)
        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            stride=1,
        )
        self.pointwise = nn.Conv2d(
            out_channels, out_channels, kernel_size=(1, 1), stride=1
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class AttnBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding, pool_size, maxpool_pad
    ):
        super(AttnBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=pool_size, padding=maxpool_pad)
        self.depthpoint = DepthwiseSeparableConv(
            in_channels, out_channels, kernel_size, padding
        )
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.maxpool(x)
        out = self.depthpoint(out)
        out = self.batchnorm(out)
        out = self.leaky_relu(out)
        return out


class AttentionModel(nn.Module):
    def __init__(self, num_classes):
        super(AttentionModel, self).__init__()
        self.main1 = MainBlock(
            in_channels=2,
            out_channels=32,
            kernel_size=(1, 3),
            padding=(0, 1),
            pool_size=(1, 2),
            maxpool_pad=0,
        )
        self.attn1 = AttnBlock(
            in_channels=2,
            out_channels=32,
            kernel_size=(1, 3),
            padding=(0, 1),
            pool_size=(1, 2),
            maxpool_pad=0,
        )
        self.main2 = MainBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 1),
            padding=(2, 0),
            pool_size=(4, 1),
            maxpool_pad=(1, 0),
        )
        self.attn2 = AttnBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 1),
            padding=(2, 0),
            pool_size=(4, 1),
            maxpool_pad=(1, 0),
        )
        self.main3 = MainBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=(1, 3),
            padding=(0, 1),
            pool_size=(1, 2),
            maxpool_pad=0,
        )
        self.attn3 = AttnBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=(1, 3),
            padding=(0, 1),
            pool_size=(1, 2),
            maxpool_pad=0,
        )
        self.main4 = MainBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=(5, 1),
            padding=(2, 0),
            pool_size=(4, 1),
            maxpool_pad=0,
        )
        self.attn4 = AttnBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=(5, 1),
            padding=(2, 0),
            pool_size=(4, 1),
            maxpool_pad=0,
        )
        self.main5 = MainBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=(5, 3),
            padding=(2, 1),
            pool_size=(4, 2),
            maxpool_pad=(1, 0),
        )
        self.attn5 = AttnBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=(5, 3),
            padding=(2, 1),
            pool_size=(4, 2),
            maxpool_pad=(1, 0),
        )
        self.main6 = MainBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=(5, 3),
            padding=(2, 1),
            pool_size=(4, 2),
            maxpool_pad=(1, 0),
        )
        self.attn6 = AttnBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=(5, 3),
            padding=(2, 1),
            pool_size=(4, 2),
            maxpool_pad=(1, 0),
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(4096, 512)
        self.leaky_relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(512, 50)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Block 1
        out_cnn = self.main1(x)
        out_attn = self.attn1(x)
        out = out_cnn * out_attn
        # Block 2
        out_cnn = self.main2(out)
        out_attn = self.attn2(out)
        out = out_cnn * out_attn
        # Block 3
        out_cnn = self.main3(out)
        out_attn = self.attn3(out)
        out = out_cnn * out_attn
        # Block 4
        out_cnn = self.main4(out)
        out_attn = self.attn4(out)
        out = out_cnn * out_attn
        # Block 5
        out_cnn = self.main5(out)
        out_attn = self.attn5(out)
        out = out_cnn * out_attn
        # Block 6
        out_cnn = self.main6(out)
        out_attn = self.attn6(out)
        out = out_cnn * out_attn
        # Last Layers
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.leaky_relu(out)
        out = self.linear2(out)
        out = self.softmax(out)
        return out
