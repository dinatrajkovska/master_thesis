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
        self.inputs = []
        self.targets = []
        print("Computing features...")
        for filename in tqdm(os.listdir(directory_path)):
            if int(filename[0]) not in dataset_folds:
                continue
            (audio, sr) = librosa.load(
                os.path.join(directory_path, filename), sr=sampling_rate
            )
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            features = []
            if log_mel:
                # https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
                log_mel_spectrogram = self.log_mel_spectrogram(audio, arguments).T
                # Normalize - min max norm
                log_mel_spectrogram = self.min_max_normalize(log_mel_spectrogram)
                log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=0)
                features.append(log_mel_spectrogram)
            if delta_log_mel:
                # https://librosa.org/doc/latest/generated/librosa.feature.delta.html
                log_mel_spectrogram = self.log_mel_spectrogram(audio, arguments)
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
            if mfcc:
                # https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
                log_mel_spectrogram = self.log_mel_spectrogram(audio, arguments)
                mel_frequency_coefficients = librosa.feature.mfcc(
                    S=log_mel_spectrogram, n_mfcc=128
                ).T
                mel_frequency_coefficients = self.min_max_normalize(
                    mel_frequency_coefficients
                )
                mel_frequency_coefficients = np.expand_dims(
                    mel_frequency_coefficients, axis=0
                )
                features.append(mel_frequency_coefficients)
            if cqt:
                # https://librosa.org/doc/latest/generated/librosa.feature.chroma_cqt.html
                # https://stackoverflow.com/questions/43838718/how-can-i-extract-cqt-from-audio-with-sampling-rate-8000hz-librosa
                constant_q = librosa.feature.chroma_cqt(
                    y=audio,
                    hop_length=arguments["hop_length"],
                    n_chroma=128,
                    bins_per_octave=128,
                ).T
                constant_q = self.min_max_normalize(constant_q)
                constant_q = np.expand_dims(constant_q, axis=0)
                features.append(constant_q)
            if chromagram:
                # https://librosa.org/doc/latest/generated/librosa.feature.chroma_stft.html
                spectrogram = librosa.stft(
                    audio, n_fft=arguments["n_fft"], hop_length=arguments["hop_length"]
                )
                chroma = librosa.feature.chroma_stft(
                    S=np.abs(spectrogram) ** 2,
                    n_fft=arguments["n_fft"],
                    hop_length=arguments["hop_length"],
                    n_chroma=128,
                ).T
                chroma = self.min_max_normalize(chroma)
                chroma = np.expand_dims(chroma, axis=0)
                features.append(chroma)

            self.inputs.append(np.concatenate(features, axis=0).astype(np.float32))

            self.targets.append(
                int(filename.split("/")[-1].split(".")[0].split("-")[-1])
            )

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def log_mel_spectrogram(self, audio, arguments):
        spectrogram = librosa.core.stft(
            audio, n_fft=arguments["n_fft"], hop_length=arguments["hop_length"]
        )
        # Convert to mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            S=np.abs(spectrogram) ** 2,
            n_fft=arguments["n_fft"],
            hop_length=arguments["hop_length"],
            n_mels=arguments["n_mels"],
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        return log_mel_spectrogram

    def min_max_normalize(self, spectrogram):
        return (spectrogram - np.min(spectrogram)) / (
            np.max(spectrogram) - np.min(spectrogram)
        )

    def __len__(self):
        return len(self.targets)
