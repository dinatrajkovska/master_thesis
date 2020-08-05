### IMPORTANT: this code might contain bugs! Go through each line of code and verify if it does the right thing!

### To be able to import the necessary packages, you can do one of two things:
### 1. Change your .bashrc file so you can import globally installed packages (such as TensorFlow)
### 2. Install the packages locally
### How to do this is described in the speech wiki. If you have questions about this, just ask me.
import os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging

from datasets import AudioDataset, target2name
from modeling import get_seq_model


def train_model(
    dataset_name,
    gammatones_path,
    sampling_rate,
    dft_window_size,
    hop_length,
    log_mel,
    delta_log_mel,
    mfcc,
    gfcc,
    cqt,
    chroma,
    learning_rate,
    weight_decay,
    batch_size,
    epochs,
    save_model_path,
    log_filepath,
):
    # Set up logging
    if log_filepath:
        logging.basicConfig(level=logging.INFO, filename=log_filepath, filemode="w")
    else:
        logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"-------- Using device {device} --------")
    ### Firstly, you need to load your data. In this example, we load the ESC-10 data set.
    ### The name of each file in the directory for this data set follows a pattern:
    ### {fold number}-{file id}-{take number}-{target class number}
    ### You can ignore take number and file id for now.
    ### The fold number indicates which test fold the file belongs to, in order to be able to perform cross-validation.
    ### You can look up information about cross-validation online.
    ### Now, for simplicity, we take folds 1-3 as train folds, 4 as validation fold and 5 as test fold.
    ### The target class number indicates which sound is present in the file.

    dataset_path = os.path.join("data", dataset_name)
    arguments = {"n_fft": dft_window_size, "hop_length": hop_length}
    logging.info("==================================")
    logging.info("Features used: ")
    logging.info(f"Log mel spectogram: {log_mel}")
    logging.info(f"Delta log mel spectogram: {delta_log_mel}")
    logging.info(f"Mel-frequency cepstral coefficients: {mfcc}")
    logging.info(f"Gammatone-frequency cepstral coefficients: {gfcc}")
    logging.info(f"Constant-Q transform: {cqt}")
    logging.info(f"STFT chromagram: {chroma}")
    logging.info("==================================")
    train_dataset = AudioDataset(
        dataset_path,
        gammatones_path,
        [1, 2, 3, 4],
        sampling_rate,
        arguments,
        log_mel,
        delta_log_mel,
        mfcc,
        gfcc,
        cqt,
        chroma,
    )

    val_dataset = AudioDataset(
        dataset_path,
        gammatones_path,
        [5],
        sampling_rate,
        arguments,
        log_mel,
        delta_log_mel,
        mfcc,
        gfcc,
        cqt,
        chroma,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    ### One option is to create a Sequential model.
    in_features = np.sum([log_mel, delta_log_mel, mfcc, gfcc, cqt, chroma])
    assert in_features > 0
    model = get_seq_model(in_features).to(device)
    # model = StupidModel(in_features)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    logging.info(
        f"Train on {len(train_dataset)}, validate on {len(val_dataset)} samples."
    )

    best_accuracy = -1
    best_epoch = -1
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch + 1}...")
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for _, audio, target in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                audio, target = audio.to(device), target.to(device)
                probs = model(audio)
                loss = criterion(probs, target)
                # backward
                loss.backward()
                # update weights
                optimizer.step()
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss.item()})

        # Set model in evaluation mode
        model.train(False)
        predictions = np.zeros(len(val_dataset))
        targets = np.zeros(len(val_dataset))
        target2total = {}
        target2correct = {}
        index = 0
        with torch.no_grad():
            for _, audio, target in tqdm(val_loader):
                audio, target = audio.to(device), target.to(device)
                probs = model(audio)
                prediction = torch.argmax(probs, dim=-1)
                # Aggregate predictions and targets
                cur_batch_size = target.size()[0]
                predictions[index : index + cur_batch_size] = prediction.cpu().numpy()
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
            if cur_accuracy > best_accuracy:
                best_accuracy = cur_accuracy
                best_epoch = epoch + 1
                logging.info("===========================")
                logging.info(
                    f"Best on epoch {epoch+1} with accuracy {best_accuracy}! Saving..."
                )
                logging.info("Per-class accuracies:")
                for target in target2total.keys():
                    logging.info(
                        f"{target2name[target]}: {target2correct[target]/target2total[target]}"
                    )
                logging.info("===========================")
                torch.save(model.state_dict(), save_model_path)
            else:
                logging.info(f"Epoch {epoch+1} with accuracy {cur_accuracy}!")

    logging.info("===========================")
    logging.info(f"Best total accuracy {best_accuracy} on epoch {best_epoch}")
    logging.info("===========================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on holdout.")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--save_model_path", default="models/best.pt", type=str)
    parser.add_argument("--mfcc", action="store_true")
    parser.add_argument("--log_mel", action="store_true")
    parser.add_argument("--delta_log_mel", action="store_true")
    parser.add_argument("--cqt", action="store_true")
    parser.add_argument("--chroma", action="store_true")
    parser.add_argument("--gfcc", action="store_true")
    parser.add_argument("--sampling_rate", default=None, type=int)
    parser.add_argument("--dataset_name", default="data_50", type=str)
    parser.add_argument(
        "--gammatones_path", default="data/gammatone_features", type=str
    )
    parser.add_argument("--dft_window_size", default=1024, type=int)
    parser.add_argument("--hop_length", default=512, type=int)
    parser.add_argument("--log_filepath", type=str, default=None)
    args = parser.parse_args()
    train_model(
        args.dataset_name,
        args.gammatones_path,
        args.sampling_rate,
        args.dft_window_size,
        args.hop_length,
        args.log_mel,
        args.delta_log_mel,
        args.mfcc,
        args.gfcc,
        args.cqt,
        args.chroma,
        args.learning_rate,
        args.weight_decay,
        args.batch_size,
        args.epochs,
        args.save_model_path,
        args.log_filepath,
    )
