### IMPORTANT: this code might contain bugs! Go through each line of code and verify if it does the right thing!

### To be able to import the necessary packages, you can do one of two things:
### 1. Change your .bashrc file so you can import globally installed packages (such as TensorFlow)
### 2. Install the packages locally
### How to do this is described in the speech wiki. If you have questions about this, just ask me.
from __future__ import division
import os
import soundfile
import numpy as np
import librosa
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


batch_size = 64
epochs = 50
sampling_rate = None
dataset_name = "data_50"
dft_window_size = 512
hop_length = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_size = {"None": 16896, "8000": 2816, "16000": 5632}
num_classes = {"data_10": 10, "data_50": 50}

### Firstly, you need to load your data. In this example, we load the ESC-10 data set.
### The name of each file in the directory for this data set follows a pattern:
### {fold number}-{file id}-{take number}-{target class number}
### You can ignore take number and file id for now.
### The fold number indicates which test fold the file belongs to, in order to be able to perform cross-validation.
### You can look up information about cross-validation online.
### Now, for simplicity, we take folds 1-3 as train folds, 4 as validation fold and 5 as test fold.
### The target class number indicates which sound is present in the file.
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, directory_path, dataset_folds)
        self.filenames = []
        for filename in os.listdir(directory_path):
            if int(filename[0]) in dataset_folds:
                self.filenames.append(os.path.join(directory_path, filename))
        self.log_mel_spectrograms = []
        self.delta_log_mel_spectrograms = []
        self.targets = []
        print("Loading data...")
        for filename in tqdm(self.filenames):
            # Read file
            (audio, sr) = librosa.load(filename, sr=sampling_rate)
            # Convert stereo to mono if necessary
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            # Extract stft
            spectrogram = librosa.core.stft(
                audio, n_fft=dft_window_size, hop_length=hop_length, center=False
            )
            # Convert to mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                S=np.abs(spectrogram) ** 2,
                n_fft=dft_window_size,
                hop_length=hop_length,
                n_mels=128,
            )
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
            log_mel_spectrogram = log_mel_spectrogram.T
            log_mel_spectrogram = np.array(log_mel_spectrogram, dtype=np.float32)
            # Normalize - min max norm
            log_mel_spectrogram = (
                log_mel_spectrogram - np.min(log_mel_spectrogram)
            ) / (np.max(log_mel_spectrogram) - np.min(log_mel_spectrogram))
            # Compute delta log mel spectrogram
            delta_log_mel_spectrogram = librosa.feature.delta(
                log_mel_spectrogram, axis=0
            )
            # Find target class number
            target_number = int(filename.split("/")[-1].split(".")[0].split("-")[-1])

            log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=0)
            delta_log_mel_spectrogram = np.expand_dims(
                delta_log_mel_spectrogram, axis=0
            )

            # Append to object vars
            self.log_mel_spectrograms.append(log_mel_spectrogram)
            self.delta_log_mel_spectrograms.append(delta_log_mel_spectrogram)
            self.targets.append(target_number)

    def __getitem__(self, idx):
        return (
            np.concatenate(
                [self.log_mel_spectrograms[idx], self.delta_log_mel_spectrograms[idx]],
                axis=0,
            ),
            self.targets[idx],
        )

    def __len__(self):
        return len(self.targets)


train_dataset = AudioDataset(dataset_name, [1, 2, 3])
val_dataset = AudioDataset(dataset_name, [4])
test_dataset = AudioDataset(dataset_name, [5])


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)§


### One option is to create a Sequential model.
model = nn.Sequential(
    # Block 1
    nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(1, 3), stride=1, padding=1),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=1, padding=1),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=1, padding=1),
    nn.BatchNorm2d(num_features=32),
    nn.MaxPool2d(kernel_size=(1, 2), padding=0),
    # Block 2
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 1), stride=1, padding=1),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 1), stride=1, padding=1),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=1, padding=1),
    nn.BatchNorm2d(num_features=32),
    nn.MaxPool2d(kernel_size=(4, 1), padding=0),
    # Block 3
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=1, padding=1),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=1),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.MaxPool2d(kernel_size=(1, 2), padding=0),
    # Block 4
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), stride=1, padding=1),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), stride=1, padding=1),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.MaxPool2d(kernel_size=(4, 1), padding=0),
    # Block 5
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 3), stride=1, padding=1),
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 3), stride=1, padding=1),
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.MaxPool2d(kernel_size=(4, 2), padding=0),
    # Block 6
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 3), stride=1, padding=1),
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 3), stride=1, padding=1),
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.MaxPool2d(kernel_size=(4, 2), padding=0),

    nn.Flatten(),
    nn.Linear(feature_size[str(sampling_rate)], 512),
    nn.LeakyReLU(),
    nn.Linear(512, num_classes[dataset_name]),
    nn.LogSoftmax(dim=1),
).to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
