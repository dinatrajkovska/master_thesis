import os
import random
from typing import List, Dict, Any

import librosa
import numpy as np
import torch
from natsort import natsorted
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from audiomentations import Compose, AddGaussianNoise, PitchShift


# https://github.com/karolpiczak/ESC-50


name2augmentation = {
    "gaussian_noise": AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    "pitch_shift": PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
}


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


class AudioDataset(TorchDataset):
    def __init__(
        self,
        directory_path,
        gammatones_path,
        dataset_folds,
        arguments,
        sampling_rate,
        log_mel,
        delta_log_mel,
        mfcc,
        gfcc,
        cqt,
        chromagram,
    ):
        self.gammatones_path = gammatones_path
        self.arguments = arguments
        self.log_mel = log_mel
        self.delta_log_mel = delta_log_mel
        self.mfcc = mfcc
        self.gfcc = gfcc
        self.cqt = cqt
        self.chromagram = chromagram
        self.sampling_rate = sampling_rate
        self.paths = []

        for filename in tqdm(natsorted(os.listdir(directory_path))):
            if int(filename[0]) not in dataset_folds:
                continue
            path = os.path.join(directory_path, filename)
            self.paths.append(path)

    def __getitem__(self, idx):
        (audio, sr) = librosa.load(self.paths[idx], sr=self.sampling_rate)
        self.arguments["sr"] = sr
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        features = []
        filename = os.path.split(self.paths[idx])[-1]

        if self.log_mel:
            # https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
            log_mel_spectrogram = self.log_mel_spectrogram(audio, self.arguments)  # .T
            # adding a dimension to be able to stack several feature channels - input to CNN
            log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=0)
            features.append(log_mel_spectrogram)
        if self.delta_log_mel:
            # https://librosa.org/doc/latest/generated/librosa.feature.delta.html
            log_mel_spectrogram = self.log_mel_spectrogram(audio, self.arguments)
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
            mel_frequency_coefficients = librosa.feature.mfcc(y=audio, n_mfcc=128).T
            mel_frequency_coefficients = np.expand_dims(
                mel_frequency_coefficients, axis=0
            )
            features.append(mel_frequency_coefficients)
        if self.gfcc:
            # https://detly.github.io/gammatone/gtgram.html
            gammatone_filename = filename.split(".")[0] + ".npy"
            gammatone = np.load(
                os.path.join(self.gammatones_path, gammatone_filename)
            ).T
            gammatone = self.min_max_normalize(gammatone)
            gammatone = np.expand_dims(gammatone, axis=0)
            features.append(gammatone)
        if self.cqt:
            # https://librosa.org/doc/latest/generated/librosa.cqt.html
            # https://librosa.org/doc/latest/generated/librosa.feature.chroma_cqt.html
            constant_q = np.abs(
                librosa.cqt(
                    audio,
                    hop_length=self.arguments["hop_length"],
                    n_bins=128,
                    bins_per_octave=128,
                )
            ).T
            constant_q = self.min_max_normalize(constant_q)
            constant_q = np.expand_dims(constant_q, axis=0)
            features.append(constant_q)
        if self.chromagram:
            # https://librosa.org/doc/latest/generated/librosa.feature.chroma_stft.html
            chroma = librosa.feature.chroma_stft(
                y=audio,
                n_fft=self.arguments["n_fft"],
                hop_length=self.arguments["hop_length"],
                n_chroma=128,
            ).T
            chroma = self.min_max_normalize(chroma)
            chroma = np.expand_dims(chroma, axis=0)
            features.append(chroma)

        return (
            self.paths[idx],
            # input
            np.concatenate(features, axis=0).astype(np.float32),
            # target
            int(filename.split(".")[0].split("-")[-1]),
        )

    def log_mel_spectrogram(self, audio, arguments):
        spectrogram = librosa.core.stft(
            audio, n_fft=arguments["n_fft"], hop_length=arguments["hop_length"]
        )
        # Convert to mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            S=np.abs(spectrogram) ** 2,
            n_fft=arguments["n_fft"],
            hop_length=arguments["hop_length"],
            n_mels=arguments["num_mels"],
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        return log_mel_spectrogram

    def __len__(self):
        return len(self.paths)


def min_max_normalize(features: np.ndarray):
    return (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-15)


def padding(sound, pad):
    # https://github.com/mil-tokyo/bc_learning_sound/blob/master/utils.py#L7
    return np.pad(sound, pad, "constant")


def multi_crop(sound, input_length, n_crops):
    # https://github.com/mil-tokyo/bc_learning_sound/blob/master/utils.py#L58
    stride = (len(sound) - input_length) // (n_crops - 1)
    sounds = [sound[stride * i : stride * i + input_length] for i in range(n_crops)]
    return np.array(sounds)


def random_crop(sound, size):
    org_size = len(sound)
    start = random.randint(0, org_size - size)
    return sound[start : start + size]


def z_score_normalize(features: np.ndarray):
    # https://www.kaggle.com/c/freesound-audio-tagging/discussion/54082
    # https://stackoverflow.com/questions/54432486/normalizing-mel-spectrogram-to-unit-peak-amplitude
    mean, std = features.mean(-1, keepdims=True), features.std(-1, keepdims=True)
    return (features - mean) / (std + 1e-15)


class PiczakBNDataset(TorchDataset):
    def __init__(
        self,
        directory_path: str,
        dataset_folds: List[str],
        train: bool,
        arguments: Dict[str, Any],
        augmentations: List[str],
    ):
        self.arguments = arguments
        self.paths = []
        self.train = train
        for filename in tqdm(natsorted(os.listdir(directory_path))):
            if filename[0] not in dataset_folds:
                continue
            path = os.path.join(directory_path, filename)
            self.paths.append(path)
        self.augmentations = Compose(
            [
                name2augmentation[name]
                for name in augmentations
                if name in name2augmentation.keys()
            ]
        )

    def __getitem__(self, idx):
        # Prepare audio basics
        audio = np.load(self.paths[idx])
        indices = np.nonzero(audio)
        first_index = indices[0].min()
        last_index = indices[0].max()
        audio = audio[first_index : last_index + 1]
        # Pad audio
        audio = padding(audio, 20480 // 2)
        features = []
        if self.train:
            # Random crop
            audio = random_crop(audio, 20480)
            # Augment audio
            audio = self.augmentations(audio, sample_rate=22050)
            log_mel = None
            if self.arguments["log_mel"]:
                log_mel = self.log_mel_spectrogram(audio, self.arguments)
                features.append(z_score_normalize(log_mel))
            if self.arguments["delta_log_mel"]:
                assert self.arguments["log_mel"]
                # https://librosa.org/doc/latest/generated/librosa.feature.delta.html
                delta_log_mel = librosa.feature.delta(log_mel, axis=0)
                features.append(z_score_normalize(delta_log_mel))
            if self.arguments["mfcc"]:
                # https://arxiv.org/abs/1706.07156 (MFCC only results on ESC-50)
                mfcc = librosa.feature.mfcc(audio, n_mfcc=self.arguments["n_features"])
                features.append(z_score_normalize(mfcc))
            if self.arguments["chroma_stft"]:
                chroma_stft = librosa.feature.chroma_stft(
                    audio, n_chroma=self.arguments["n_features"], norm=None
                )
                features.append(z_score_normalize(chroma_stft))
            if self.arguments["chroma_cqt"]:
                chroma_cqt = librosa.feature.chroma_cqt(
                    audio,
                    bins_per_octave=self.arguments["n_features"],
                    n_chroma=self.arguments["n_features"],
                    norm=None,
                )
                features.append(z_score_normalize(chroma_cqt))
            features = np.concatenate(
                [np.expand_dims(feature, axis=0) for feature in features], axis=0
            ).astype(np.float32)
        else:
            # Multi-crop audio
            audio = multi_crop(audio, 20480, 10)
            for i in range(10):
                crop_features = []
                log_mel = None
                if self.arguments["log_mel"]:
                    log_mel = self.log_mel_spectrogram(audio[i], self.arguments)
                    crop_features.append(z_score_normalize(log_mel))
                if self.arguments["delta_log_mel"]:
                    assert self.arguments["log_mel"]
                    # https://librosa.org/doc/latest/generated/librosa.feature.delta.html
                    delta_log_mel = librosa.feature.delta(log_mel, axis=0)
                    crop_features.append(z_score_normalize(delta_log_mel))
                if self.arguments["mfcc"]:
                    # https://arxiv.org/abs/1706.07156 (MFCC only results on ESC-50)
                    mfcc = librosa.feature.mfcc(
                        audio[i], n_mfcc=self.arguments["n_features"]
                    )
                    crop_features.append(z_score_normalize(mfcc))
                if self.arguments["chroma_stft"]:
                    chroma_stft = librosa.feature.chroma_stft(
                        audio[i], n_chroma=self.arguments["n_features"], norm=None
                    )
                    crop_features.append(z_score_normalize(chroma_stft))
                if self.arguments["chroma_cqt"]:
                    chroma_cqt = librosa.feature.chroma_cqt(
                        audio[i],
                        bins_per_octave=self.arguments["n_features"],
                        n_chroma=self.arguments["n_features"],
                        norm=None,
                    )
                    crop_features.append(z_score_normalize(chroma_cqt))
                crop_features = np.concatenate(
                    [np.expand_dims(feature, axis=0) for feature in crop_features],
                    axis=0,
                )
                features.append(crop_features)
            features = np.stack(features, axis=0)

        filename = os.path.split(self.paths[idx])[-1]
        label = int(filename.split(".")[0].split("-")[-1])

        return (
            # input
            features,
            # target
            label,
        )

    def log_mel_spectrogram(self, audio, arguments):
        # https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
        # Convert to mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            audio,
            n_fft=arguments["n_fft"],
            hop_length=arguments["hop_length"],
            n_mels=arguments["n_features"],
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        return log_mel_spectrogram

    def __len__(self):
        return len(self.paths)


class EnvNetDataset(TorchDataset):
    def __init__(self, dataset_path: str, dataset_folds: List[int], train: bool = True):
        self.train = train
        self.paths = []
        for filename in tqdm(natsorted(os.listdir(dataset_path))):
            if int(filename[0]) not in dataset_folds:
                continue
            path = os.path.join(dataset_path, filename)
            self.paths.append(path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        # Obtain input
        audio = np.load(self.paths[idx])
        indices = np.nonzero(audio)
        first_index = indices[0].min()
        last_index = indices[0].max()
        audio = audio[first_index : last_index + 1]
        # Pad audio
        audio = padding(audio, 20480 // 2)
        if self.train:
            # Random crop
            audio = random_crop(audio, 20480)
            # Prepare audio
            audio = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
        else:
            # Multi-crop audio
            audio = multi_crop(audio, 20480, 10)
            # Prepare audio
            audio = torch.from_numpy(audio).unsqueeze(1).unsqueeze(1)

        filename = os.path.split(self.paths[idx])[-1]
        label = int(filename.split(".")[0].split("-")[-1])

        return audio, label
