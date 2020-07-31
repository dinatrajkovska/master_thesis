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
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import os

from datasets import AudioDataset
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
):
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
    arguments = {"n_fft": dft_window_size, "hop_length": hop_length, "n_mels": 128}
    print("==================================")
    print("Features used: ")
    print(f"Log mel spectogram: {log_mel}")
    print(f"Delta log mel spectogram: {delta_log_mel}")
    print(f"Mel-frequency cepstral coefficients: {mfcc}")
    print(f"Constant-Q transform: {cqt}")
    print(f"STFT chromagram: {chroma}")
    print("==================================")
    dataset_path = os.path.join("data", dataset_name)
    for split_num, split in enumerate(data_splits):
        print(f"----------- Starting split number {split_num + 1} -----------")
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

        print(
            f"Train on {len(train_dataset)}, validate on {len(test_dataset)} samples."
        )

        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}...")
            # Set model in train mode
            model.train(True)
            with tqdm(total=len(train_loader)) as pbar:
                for audio, target in train_loader:
                    # remove past gradients
                    optimizer.zero_grad()
                    # forward
                    audio, target = audio.to(device), target.to(device)
                    prediction = model(audio)
                    loss = criterion(prediction, target)
                    # backward
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({"Batch loss": loss.item()})

        # Set model in evaluation mode
        model.train(False)
        predictions = np.zeros(len(test_dataset))
        targets = np.zeros(len(test_dataset))
        index = 0
        with torch.no_grad():
            for audio, target in tqdm(test_loader):
                audio, target = audio.to(device), target.to(device)
                probs = model(audio)
                pred = probs.argmax(dim=-1)
                # Aggregate predictions and targets
                cur_batch_size = target.size()[0]
                predictions[index : index + cur_batch_size] = pred.cpu().numpy()
                targets[index : index + cur_batch_size] = target.cpu().numpy()
                index += cur_batch_size

            print(f"Test accuracy: {accuracy_score(targets, predictions)}!")
            total_accuracy += accuracy_score(targets, predictions)

    print(f"The total accuracy is {total_accuracy / len(data_splits)}")


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
    )
