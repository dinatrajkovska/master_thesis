from torch import nn


def get_seq_model(feature_size, num_classes):
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
