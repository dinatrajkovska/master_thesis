import math
import os
import random
from typing import Any, Dict, List

import librosa
import numpy as np
import torch
from audiomentations import AddGaussianNoise, Compose, PitchShift
from gammatone.fftweight import fft_gtgram
from natsort import natsorted
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

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
    return sound[start : start + size], start, org_size


def z_score_normalize(features: np.ndarray):
    # https://www.kaggle.com/c/freesound-audio-tagging/discussion/54082
    # https://stackoverflow.com/questions/54432486/normalizing-mel-spectrogram-to-unit-peak-amplitude
    mean, std = features.mean(-1, keepdims=True), features.std(-1, keepdims=True)
    return (features - mean) / (std + 1e-15)


def random_cqt_gfcc_crop(feature, start_audio, size_audio):
    start_f = math.floor((start_audio / size_audio) * feature.shape[1])
    end_f = start_f + 41  # Fixed because crop-size is fixed
    return feature[:, start_f:end_f]


class PiczakBNDataset(TorchDataset):
    def __init__(
        self,
        directory_path: str,
        dataset_folds: List[str],
        train: bool,
        arguments: Dict[str, Any],
        augmentations: List[str],
        train_cqts_path: str = None,
        train_gfccs_path: str = None,
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
        self.train_cqts_path = train_cqts_path
        self.train_gfccs_path = train_gfccs_path
        if self.arguments["cqt"]:
            assert (
                self.train_cqts_path is not None
            ), "cqt feature is True but there is no train_cqts_path"
        if self.arguments["gfcc"]:
            assert (
                self.train_gfccs_path is not None
            ), "gfcc feature is True but there is no train_gfccs_path"

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
        audio_name = os.path.split(self.paths[idx])[-1].split(".")[0]
        if self.train:
            # Random crop
            audio, start_audio, size_audio = random_crop(audio, 20480)
            # Augment audio
            audio = self.augmentations(
                audio, sample_rate=self.arguments["sampling_rate"]
            )
            log_mel = None
            if self.arguments["log_mel"]:
                log_mel = self.log_mel_spectrogram(audio)
                features.append(z_score_normalize(log_mel))
            if self.arguments["delta_log_mel"]:
                assert self.arguments["log_mel"]
                # https://librosa.org/doc/latest/generated/librosa.feature.delta.html
                delta_log_mel = librosa.feature.delta(log_mel, axis=0)
                features.append(z_score_normalize(delta_log_mel))
            if self.arguments["mfcc"]:
                # https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
                # https://arxiv.org/abs/1706.07156 (MFCC only results on ESC-50)
                mfcc = librosa.feature.mfcc(
                    audio,
                    n_mfcc=self.arguments["n_features"],
                    sr=self.arguments["sampling_rate"],
                )
                features.append(z_score_normalize(mfcc))
            if self.arguments["gfcc"]:
                # https://detly.github.io/gammatone/fftweight.html
                gfcc_path = os.path.join(
                    self.train_gfccs_path, f"{audio_name}-gfcc.npy"
                )
                gfcc = np.load(gfcc_path)
                gfcc = random_cqt_gfcc_crop(gfcc, start_audio, size_audio)
                features.append(z_score_normalize(gfcc))
            if self.arguments["chroma_stft"]:
                # https://librosa.org/doc/latest/generated/librosa.feature.chroma_stft.html
                chroma_stft = librosa.feature.chroma_stft(
                    audio,
                    n_chroma=self.arguments["n_features"],
                    norm=None,
                    sr=self.arguments["sampling_rate"],
                    n_fft=self.arguments["n_fft"],
                    hop_length=self.arguments["hop_length"],
                )
                features.append(z_score_normalize(chroma_stft))
            if self.arguments["cqt"]:
                # https://librosa.org/doc/latest/generated/librosa.cqt.html#librosa.cqt
                cqt_path = os.path.join(self.train_cqts_path, f"{audio_name}-cqt.npy")
                cqt = np.abs(np.load(cqt_path))
                cqt = random_cqt_gfcc_crop(cqt, start_audio, size_audio)
                features.append(z_score_normalize(cqt))
            features = np.concatenate(
                [np.expand_dims(feature, axis=0) for feature in features], axis=0
            ).astype(np.float32)
        else:
            # Multi-crop audio
            audio_name = os.path.split(self.paths[idx])[-1].split(".")[0]
            audio = multi_crop(audio, 20480, 10)
            for i in range(10):
                crop_features = []
                log_mel = None
                if self.arguments["log_mel"]:
                    log_mel = self.log_mel_spectrogram(audio[i])
                    crop_features.append(z_score_normalize(log_mel))
                if self.arguments["delta_log_mel"]:
                    assert self.arguments["log_mel"]
                    # https://librosa.org/doc/latest/generated/librosa.feature.delta.html
                    delta_log_mel = librosa.feature.delta(log_mel, axis=0)
                    crop_features.append(z_score_normalize(delta_log_mel))
                if self.arguments["mfcc"]:
                    # https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
                    # https://arxiv.org/abs/1706.07156 (MFCC only results on ESC-50)
                    mfcc = librosa.feature.mfcc(
                        audio[i],
                        n_mfcc=self.arguments["n_features"],
                        sr=self.arguments["sampling_rate"],
                    )
                    crop_features.append(z_score_normalize(mfcc))
                if self.arguments["gfcc"]:
                    # https://detly.github.io/gammatone/fftweight.html
                    gfcc = fft_gtgram(
                        wave=audio[i],
                        fs=self.arguments["sampling_rate"],
                        window_time=self.arguments["n_fft"]
                        / self.arguments["sampling_rate"],
                        hop_time=(self.arguments["hop_length"] - 52)
                        / self.arguments["sampling_rate"],
                        channels=self.arguments["n_features"],
                        f_min=32.7,
                    )
                    crop_features.append(z_score_normalize(gfcc))
                if self.arguments["chroma_stft"]:
                    # https://librosa.org/doc/latest/generated/librosa.feature.chroma_stft.html
                    chroma_stft = librosa.feature.chroma_stft(
                        audio[i],
                        n_chroma=self.arguments["n_features"],
                        norm=None,
                        sr=self.arguments["sampling_rate"],
                        n_fft=self.arguments["n_fft"],
                        hop_length=self.arguments["hop_length"],
                    )
                    crop_features.append(z_score_normalize(chroma_stft))
                if self.arguments["cqt"]:
                    # https://librosa.org/doc/latest/generated/librosa.cqt.html#librosa.cqt
                    cqt = np.abs(
                        librosa.cqt(
                            audio[i],
                            sr=self.arguments["sampling_rate"],
                            hop_length=self.arguments["hop_length"],
                            n_bins=self.arguments["n_features"],
                            norm=None,
                        )
                    )
                    crop_features.append(z_score_normalize(cqt))
                crop_features = np.concatenate(
                    [np.expand_dims(feature, axis=0) for feature in crop_features],
                    axis=0,
                )
                features.append(crop_features)
            features = np.stack(features, axis=0).astype(np.float32)

        filename = os.path.split(self.paths[idx])[-1]
        label = int(filename.split(".")[0].split("-")[-1])

        return features, label

    def log_mel_spectrogram(self, audio):
        # https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
        mel_spectrogram = librosa.feature.melspectrogram(
            audio,
            n_fft=self.arguments["n_fft"],
            hop_length=self.arguments["hop_length"],
            n_mels=self.arguments["n_features"],
            sr=self.arguments["sampling_rate"],
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        return log_mel_spectrogram

    def __len__(self):
        return len(self.paths)


class EnvNetDataset(TorchDataset):
    def __init__(
        self,
        dataset_path: str,
        dataset_folds: List[int],
        train: bool,
        augmentations: List[str],
    ):
        self.train = train
        self.paths = []
        for filename in tqdm(natsorted(os.listdir(dataset_path))):
            if filename[0] not in dataset_folds:
                continue
            path = os.path.join(dataset_path, filename)
            self.paths.append(path)
        self.sampling_rate = int(dataset_path.split("_")[-1])
        self.augmentations = Compose(
            [
                name2augmentation[name]
                for name in augmentations
                if name in name2augmentation.keys()
            ]
        )

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
        audio = padding(audio, 66650 // 2)
        if self.train:
            # Random crop
            audio, _, _ = random_crop(audio, 66650)
            # Augment audio
            audio = self.augmentations(audio, sample_rate=self.sampling_rate)
            # Prepare audio
            audio = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
        else:
            # Multi-crop audio
            audio = multi_crop(audio, 66650, 10)
            # Prepare audio
            audio = torch.from_numpy(audio).unsqueeze(1).unsqueeze(1)

        filename = os.path.split(self.paths[idx])[-1]
        label = int(filename.split(".")[0].split("-")[-1])

        return audio, label
