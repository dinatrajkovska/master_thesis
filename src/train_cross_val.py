### IMPORTANT: this code might contain bugs! Go through each line of code and verify if it does the right thing!

### To be able to import the necessary packages, you can do one of two things:
### 1. Change your .bashrc file so you can import globally installed packages (such as TensorFlow)
### 2. Install the packages locally
### How to do this is described in the speech wiki. If you have questions about this, just ask me.
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import logging

from datasets import AudioDataset, target2name
from modeling import get_seq_model


def train_model(
    dataset_name,
    sampling_rate,
    dft_window_size,
    hop_length,
    log_mel,
    delta_log_mel,
    mfcc,
    cqt,
    chroma,
    learning_rate,
    batch_size,
    epochs,
    log_filepath,
):
    # Set up logging
    if log_filepath:
        logging.basicConfig(level=logging.INFO, filename=log_filepath, filemode="w")
    else:
        logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### Firstly, you need to load your data. In this example, we load the ESC-10 data set.
    ### The name of each file in the directory for this data set follows a pattern:
    ### {fold number}-{file id}-{take number}-{target class number}
    ### You can ignore take number and file id for now.
    ### The fold number indicates which test fold the file belongs to, in order to be able to perform cross-validation.
    ### You can look up information about cross-validation online.
    ### Now, for simplicity, we take folds 1-3 as train folds, 4 as validation fold and 5 as test fold.
    ### The target class number indicates which sound is present in the file.

    data_splits = [
        ([1, 2, 3, 4], [5]),
        ([1, 2, 3, 5], [4]),
        ([1, 2, 4, 5], [3]),
        ([1, 3, 4, 5], [2]),
        ([2, 3, 4, 5], [1]),
    ]
    total_accuracy = 0
    total_per_class = {}
    arguments = {"n_fft": dft_window_size, "hop_length": hop_length, "n_mels": 128}
    logging.info("==================================")
    logging.info("Features used: ")
    logging.info(f"Log mel spectogram: {log_mel}")
    logging.info(f"Delta log mel spectogram: {delta_log_mel}")
    logging.info(f"Mel-frequency cepstral coefficients: {mfcc}")
    logging.info(f"Constant-Q transform: {cqt}")
    logging.info(f"STFT chromagram: {chroma}")
    logging.info("==================================")
    dataset_path = os.path.join("data", dataset_name)
    for split_num, split in enumerate(data_splits):
        logging.info(f"----------- Starting split number {split_num + 1} -----------")
        train_dataset = AudioDataset(
            dataset_path,
            split[0],
            sampling_rate,
            arguments,
            log_mel,
            delta_log_mel,
            mfcc,
            cqt,
            chroma,
        )
        test_dataset = AudioDataset(
            dataset_path,
            split[1],
            sampling_rate,
            arguments,
            log_mel,
            delta_log_mel,
            mfcc,
            cqt,
            chroma,
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        ### One option is to create a Sequential model.
        in_features = np.sum([log_mel, delta_log_mel, mfcc, cqt, chroma])
        assert in_features > 0
        model = get_seq_model(in_features).to(device)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        logging.info(
            f"Train on {len(train_dataset)}, validate on {len(test_dataset)} samples."
        )

        for epoch in range(epochs):
            # Set model in train mode
            model.train(True)
            for audio, target in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                audio, target = audio.to(device), target.to(device)
                prediction = model(audio)
                # L2 regularization on the penultimate dense layer
                loss = criterion(prediction, target) + model[31].weight.norm(2) * 0.1
                # backward
                loss.backward()
                # update weights
                optimizer.step()

        # Set model in evaluation mode
        model.train(False)
        predictions = np.zeros(len(test_dataset))
        targets = np.zeros(len(test_dataset))
        target2total = {}
        target2correct = {}
        index = 0
        with torch.no_grad():
            for audio, target in test_loader:
                audio, target = audio.to(device), target.to(device)
                probs = model(audio)
                pred = probs.argmax(dim=-1)
                # Aggregate predictions and targets
                cur_batch_size = target.size()[0]
                predictions[index : index + cur_batch_size] = pred.cpu().numpy()
                targets[index : index + cur_batch_size] = target.cpu().numpy()
                index += cur_batch_size

            for i in range(predictions.shape[0]):
                target = targets[i]
                pred = predictions[i]
                if target not in target2total:
                    target2total[target] = 0
                    target2correct[target] = 0
                target2total[target] += 1
                if pred == target:
                    target2correct[target] += 1

            cur_accuracy = sum(target2correct.values()) / sum(target2total.values())
            logging.info(f"Test accuracy: {cur_accuracy}!")
            logging.info("Per-class accuracies:")
            for target in target2total.keys():
                # Obtain class name and class accuracy
                class_name = target2name[target]
                class_accuracy = target2correct[target] / target2total[target]
                logging.info(f"{class_name}: {class_accuracy}")
                # Aggregate per class accuracies - happens only the first time
                if class_name not in total_per_class:
                    total_per_class[class_name] = 0
                total_per_class[class_name] += class_accuracy
            total_accuracy += cur_accuracy

    logging.info("=================")
    logging.info("The averaged per-class accuracies are: ")
    for class_name, class_accuracy in total_per_class.items():
        logging.info(f"{class_name}: {class_accuracy / len(data_splits)}")
    logging.info(f"The averaged accuracy is {total_accuracy / len(data_splits)}")
    logging.info("================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--sampling_rate", default=None, type=int)
    parser.add_argument("--dataset_name", default="data_50", type=str)
    parser.add_argument("--dft_window_size", default=1024, type=int)
    parser.add_argument("--hop_length", default=512, type=int)
    parser.add_argument("--mfcc", action="store_true")
    parser.add_argument("--cqt", action="store_true")
    parser.add_argument("--chroma", action="store_true")
    parser.add_argument("--log_mel", action="store_true")
    parser.add_argument("--delta_log_mel", action="store_true")
    parser.add_argument("--log_filepath", type=str, default=None)

    args = parser.parse_args()
    train_model(
        args.dataset_name,
        args.sampling_rate,
        args.dft_window_size,
        args.hop_length,
        args.log_mel,
        args.delta_log_mel,
        args.mfcc,
        args.cqt,
        args.chroma,
        args.learning_rate,
        args.batch_size,
        args.epochs,
        args.log_filepath,
    )
