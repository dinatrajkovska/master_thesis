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
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import argparse

from datasets import AudioDataset
from modeling import get_seq_model


def train_model(
    batch_size,
    epochs,
    save_model_path,
    sampling_rate,
    dataset_name,
    dft_window_size,
    hop_length,
):
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

    dataset_path = os.path.join("data", dataset_name)
    train_dataset = Subset(
        AudioDataset(
            dataset_path, [1, 2, 3], sampling_rate, dft_window_size, hop_length
        ),
        list(range(10)),
    )
    val_dataset = Subset(
        AudioDataset(dataset_path, [4], sampling_rate, dft_window_size, hop_length),
        list(range(5)),
    )
    test_dataset = Subset(
        AudioDataset(dataset_path, [5], sampling_rate, dft_window_size, hop_length),
        list(range(5)),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    ### One option is to create a Sequential model.
    model = get_seq_model(feature_size, num_classes[dataset_name]).to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    print(f"Train on {len(train_dataset)}, validate on {len(val_dataset)} samples.")

    best_accuracy = -1
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
        predictions = np.zeros(len(val_dataset))
        targets = np.zeros(len(val_dataset))
        index = 0
        with torch.no_grad():
            for audio, target in tqdm(val_loader):
                audio, target = audio.to(device), target.to(device)
                probs = model(audio)
                pred = probs.argmax(dim=-1)
                # Aggregate predictions and targets
                cur_batch_size = target.size()[0]
                predictions[index : index + cur_batch_size] = pred.cpu().numpy()
                targets[index : index + cur_batch_size] = target.cpu().numpy()
                index += cur_batch_size

            cur_accuracy = accuracy_score(targets, predictions)
            if cur_accuracy > best_accuracy:
                best_accuracy = cur_accuracy
                print("===========================")
                print(
                    f"Best on epoch {epoch+1} with accuracy {best_accuracy}! Saving..."
                )
                print("===========================")
                torch.save(model.state_dict(), save_model_path)
            else:
                print(f"Epoch {epoch+1} with accuracy {cur_accuracy}!")

    model.load_state_dict(torch.load(save_model_path))
    # Set model in evaluation mode
    model.train(False)
    predictions = np.zeros(len(val_dataset))
    targets = np.zeros(len(val_dataset))
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

        print(f"Test accuracy {accuracy_score(targets, predictions)}!")
        precisions, recalls, _, _ = precision_recall_fscore_support(
            targets, predictions, zero_division=0
        )
        for i, (precision, recall) in enumerate(zip(precisions, recalls)):
            print(f"The precision | recall for class {i}: {precision} | {recall}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--save_model_path", default="models/best.pt", type=str)
    parser.add_argument("--sampling_rate", default=None, type=int)
    parser.add_argument("--dataset_name", default="data_50", type=str)
    parser.add_argument("--dft_window_size", default=512, type=int)
    parser.add_argument("--hop_length", default=512, type=int)
    args = parser.parse_args()
    train_model(
        args.batch_size,
        args.epochs,
        args.save_model_path,
        args.sampling_rate,
        args.dataset_name,
        args.dft_window_size,
        args.hop_length,
    )
