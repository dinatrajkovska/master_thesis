import torch
import numpy as np
from tqdm import tqdm
import librosa
import os


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self, directory_path, dataset_folds, sampling_rate, dft_window_size, hop_length
    ):
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
