import torch
import numpy as np
from tqdm import tqdm
import librosa
import os


# https://github.com/karolpiczak/ESC-50


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        directory_path,
        dataset_folds,
        sampling_rate,
        arguments,
        log_mel,
        delta_log_mel,
        mfcc,
        cqt,
        chromagram,
    ):
        self.audios = []
        self.targets = []
        for filename in tqdm(os.listdir(directory_path)):
            if int(filename[0]) not in dataset_folds:
                continue
            (audio, sr) = librosa.load(
                os.path.join(directory_path, filename), sr=sampling_rate
            )
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            self.audios.append(audio)
            self.targets.append(
                int(filename.split("/")[-1].split(".")[0].split("-")[-1])
            )
        self.arguments = arguments
        self.log_mel = log_mel
        self.delta_log_mel = delta_log_mel
        self.mfcc = mfcc
        self.cqt = cqt
        self.chromagram = chromagram

    def __getitem__(self, idx):
        audio = self.audios[idx]
        # Extract stft
        features = []
        if self.log_mel:
            log_mel_spectrogram = self.log_mel_spectrogram(audio)
            # Normalize - min max norm
            log_mel_spectrogram = self.min_max_normalize(log_mel_spectrogram)
            log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=0)
            features.append(log_mel_spectrogram)
        if self.delta_log_mel:
            log_mel_spectrogram = self.log_mel_spectrogram(audio)
            # Normalize - min max norm
            log_mel_spectrogram = self.min_max_normalize(log_mel_spectrogram)
            # Compute delta log mel spectrogram
            delta_log_mel_spectrogram = librosa.feature.delta(
                log_mel_spectrogram, axis=0
            )
            # Expand dims
            delta_log_mel_spectrogram = np.expand_dims(
                delta_log_mel_spectrogram, axis=0
            )
            features.append(delta_log_mel_spectrogram)
        if self.mfcc:
            # https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
            mel_frequency_coefficients = librosa.feature.mfcc(
                y=audio, n_mfcc=128, **self.arguments
            )
            mel_frequency_coefficients = self.min_max_normalize(
                mel_frequency_coefficients
            )
            mel_frequency_coefficients = mel_frequency_coefficients.T
            mel_frequency_coefficients = np.expand_dims(
                mel_frequency_coefficients, axis=0
            )
            features.append(mel_frequency_coefficients)
        if self.cqt:
            # https://librosa.org/doc/latest/generated/librosa.feature.chroma_cqt.html
            # https://stackoverflow.com/questions/43838718/how-can-i-extract-cqt-from-audio-with-sampling-rate-8000hz-librosa
            constant_q = librosa.feature.chroma_cqt(
                y=audio, hop_length=512, n_chroma=128, bins_per_octave=128
            )
            constant_q = constant_q.T
            constant_q = self.min_max_normalize(constant_q)
            constant_q = np.expand_dims(constant_q, axis=0)
            features.append(constant_q)
        if self.chromagram:
            # https://librosa.org/doc/latest/generated/librosa.feature.chroma_stft.html
            spectrogram = librosa.core.stft(
                audio,
                n_fft=self.arguments["n_fft"],
                hop_length=self.arguments["hop_length"],
                center=self.arguments["center"],
            )
            chroma = librosa.feature.chroma_stft(
                S=np.abs(spectrogram) ** 2,
                n_fft=self.arguments["n_fft"],
                hop_length=self.arguments["hop_length"],
                n_chroma=self.arguments["n_mels"],
            )
            chroma = chroma.T
            chroma = self.min_max_normalize(chroma)
            chroma = np.expand_dims(chroma, axis=0)
            features.append(chroma)

        return np.concatenate(features, axis=0).astype(np.float32), self.targets[idx]

    def log_mel_spectrogram(self, audio):
        spectrogram = librosa.core.stft(
            audio,
            n_fft=self.arguments["n_fft"],
            hop_length=self.arguments["hop_length"],
            center=self.arguments["center"],
        )
        # Convert to mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            S=np.abs(spectrogram) ** 2,
            n_fft=self.arguments["n_fft"],
            hop_length=self.arguments["hop_length"],
            n_mels=self.arguments["n_mels"],
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        return log_mel_spectrogram.T

    def min_max_normalize(self, spectrogram):
        return (spectrogram - np.min(spectrogram)) / (
            np.max(spectrogram) - np.min(spectrogram)
        )

    def __len__(self):
        return len(self.targets)
