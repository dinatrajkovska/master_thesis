from torch import nn


def get_seq_model(num_classes):
    return nn.Sequential(
        # Block 1
        nn.Conv2d(
            in_channels=2, out_channels=32, kernel_size=(1, 3), stride=1, padding=(0, 1)
        ),
        nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        ),
        nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=(1, 1), stride=1, padding=0
        ),
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
        nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0),
        ),
        nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=(1, 1), stride=1, padding=0
        ),
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
        nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        ),
        nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(1, 1), stride=1, padding=0
        ),
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
        nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0),
        ),
        nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(1, 1), stride=1, padding=0
        ),
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
        nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(5, 3),
            stride=1,
            padding=(2, 1),
        ),
        nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1, padding=0
        ),
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
        nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(5, 3),
            stride=1,
            padding=(2, 1),
        ),
        nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(1, 1), stride=1, padding=0
        ),
        nn.BatchNorm2d(num_features=256),
        nn.MaxPool2d(kernel_size=(4, 2), padding=(1, 0)),
        nn.Flatten(),
        nn.Linear(4096, 512),
        nn.LeakyReLU(),
        nn.Linear(512, num_classes),
        nn.LogSoftmax(dim=1),
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
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=pool_size, padding=maxpool_pad)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
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
        self, in_channels, out_channels, kernel_size, padding, pool_size, maxpool_pad, 
    ):
        super(AttnBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=pool_size, padding=maxpool_pad)
        self.depthpoint = DepthwiseSeparableConv(
            in_channels, out_channels, kernel_size, padding
        )
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.maxpool(x)
        out = self.depthpoint(out)
        out = self.batchnorm(out)
        out = self.relu(out)
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
        self.linear2 = nn.Linear(512, num_classes)
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
