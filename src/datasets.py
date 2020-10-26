import torch
import numpy as np
from tqdm import tqdm
import librosa
import os

# https://github.com/karolpiczak/ESC-50


target2name = {
    0: "dog",
    1: "rooster",
    2: "pig",
    3: "cow",
    4: "frog",
    5: "cat",
    6: "hen",
    7: "insects",
    8: "sheep",
    9: "crow",
    10: "rain",
    11: "sea_waves",
    12: "crackling_fire",
    13: "crickets",
    14: "chirping_birds",
    15: "water_drops",
    16: "wind",
    17: "pouring_water",
    18: "toilet_flush",
    19: "thunderstorm",
    20: "crying_baby",
    21: "sneezing",
    22: "clapping",
    23: "breathing",
    24: "coughing",
    25: "footsteps",
    26: "laughing",
    27: "brushing_teeth",
    28: "snoring",
    29: "drinking_sipping",
    30: "door_wood_knock",
    31: "mouse_click",
    32: "keyboard_typing",
    33: "door_wood_creaks",
    34: "can_opening",
    35: "washing_machine",
    36: "vacuum_cleaner",
    37: "clock_alarm",
    38: "clock_tick",
    39: "glass_breaking",
    40: "helicopter",
    41: "chainsaw",
    42: "siren",
    43: "car_horn",
    44: "engine",
    45: "train",
    46: "church_bells",
    47: "airplane",
    48: "fireworks",
    49: "hand_saw",
}


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        directory_path,
        gammatones_path,
        dataset_folds,
        sampling_rate,
        arguments,
        log_mel,
        delta_log_mel,
        mfcc,
        gfcc,
        cqt,
        chromagram,
    ):
        self.inputs = []
        self.targets = []
        self.paths = []
        for filename in tqdm(os.listdir(directory_path)):
            if int(filename[0]) not in dataset_folds:
                continue
            path = os.path.join(directory_path, filename)
            (audio, sr) = librosa.load(path, sr=sampling_rate)
            arguments["sr"] = sr
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            features = []
            if log_mel:
                # https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
                log_mel_spectrogram = self.log_mel_spectrogram(audio, arguments)#.T
                # Normalize - min max norm
                log_mel_spectrogram = self.min_max_normalize(log_mel_spectrogram)
                # adding a dimension to be able to stack several feature channels - input to CNN
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
                mel_frequency_coefficients = librosa.feature.mfcc(
                    y=audio, n_mfcc=128, sr=arguments["sr"]
                ).T
                mel_frequency_coefficients = self.min_max_normalize(
                    mel_frequency_coefficients
                )
                mel_frequency_coefficients = np.expand_dims(
                    mel_frequency_coefficients, axis=0
                )
                features.append(mel_frequency_coefficients)
            if gfcc:
                # https://detly.github.io/gammatone/gtgram.html
                gammatone_filename = filename.split(".")[0] + ".npy"
                gammatone = np.load(os.path.join(gammatones_path, gammatone_filename)).T
                gammatone = self.min_max_normalize(gammatone)
                gammatone = np.expand_dims(gammatone, axis=0)
                features.append(gammatone)
            if cqt:
                # https://librosa.org/doc/latest/generated/librosa.cqt.html
                # https://librosa.org/doc/latest/generated/librosa.feature.chroma_cqt.html
                # constant_q = librosa.feature.chroma_cqt(
                #     y=audio,
                #     sr=arguments["sr"],
                #     hop_length=arguments["hop_length"],
                #     n_chroma=128,
                #     bins_per_octave=128,
                # ).T
                constant_q = np.abs(
                    librosa.cqt(
                        audio,
                        sr=arguments["sr"],
                        hop_length=arguments["hop_length"],
                        n_bins=128,
                        bins_per_octave=128,
                    )
                ).T
                constant_q = self.min_max_normalize(constant_q)
                constant_q = np.expand_dims(constant_q, axis=0)
                features.append(constant_q)
            if chromagram:
                # https://librosa.org/doc/latest/generated/librosa.feature.chroma_stft.html
                chroma = librosa.feature.chroma_stft(
                    y=audio,
                    sr=arguments["sr"],
                    n_fft=arguments["n_fft"],
                    hop_length=arguments["hop_length"],
                    n_chroma=128,
                ).T
                chroma = self.min_max_normalize(chroma)
                chroma = np.expand_dims(chroma, axis=0)
                features.append(chroma)

            self.inputs.append(np.concatenate(features, axis=0).astype(np.float32))
            self.paths.append(path)
            self.targets.append(int(filename.split(".")[0].split("-")[-1]))

    def __getitem__(self, idx):
        return self.paths[idx], self.inputs[idx], self.targets[idx]

    def log_mel_spectrogram(self, audio, arguments):
        spectrogram = librosa.core.stft(
            audio, n_fft=arguments["n_fft"], hop_length=arguments["hop_length"]
        )
        # Convert to mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            S=np.abs(spectrogram) ** 2,
            sr=arguments["sr"],
            n_fft=arguments["n_fft"],
            hop_length=arguments["hop_length"],
            n_mels=arguments["num_mels"],
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        return log_mel_spectrogram

    def min_max_normalize(self, X):
        # normalization_factor = 1 / np.max(np.abs(clip))
        # clip = clip * normalization_factor
        return (X - np.min(X)) / (np.max(X) - np.min(X))

    def __len__(self):
        return len(self.targets)
