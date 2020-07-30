### IMPORTANT: this code might contain bugs! Go through each line of code and verify if it does the right thing!

### To be able to import the necessary packages, you can do one of two things:
### 1. Change your .bashrc file so you can import globally installed packages (such as TensorFlow)
### 2. Install the packages locally
### How to do this is described in the speech wiki. If you have questions about this, just ask me.
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import argparse

from datasets import AudioDataset
from modeling import get_seq_model


def inference(
    batch_size,
    checkpoint_path,
    sampling_rate,
    dataset_name,
    dft_window_size,
    hop_length,
    log_mel,
    delta_log_mel,
    mfcc,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-------- Using device {device} --------")
    ### Firstly, you need to load your data. In this example, we load the ESC-10 data set.
    ### The name of each file in the directory for this data set follows a pattern:
    ### {fold number}-{file id}-{take number}-{target class number}
    ### You can ignore take number and file id for now.
    ### The fold number indicates which test fold the file belongs to, in order to be able to perform cross-validation.
    ### You can look up information about cross-validation online.
    ### Now, for simplicity, we take folds 1-3 as train folds, 4 as validation fold and 5 as test fold.
    ### The target class number indicates which sound is present in the file.

    dataset_path = os.path.join("data", dataset_name)
    arguments = {
        "n_fft": dft_window_size,
        "hop_length": hop_length,
        "n_mels": 128,
        "center": False,
    }
    test_dataset = AudioDataset(dataset_path, [5], sampling_rate, arguments)
    print(f"Inference on {len(test_dataset)} samples.")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    # Model
    in_features = np.sum([log_mel, delta_log_mel, mfcc])
    model = get_seq_model(in_features).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    # Set model in evaluation mode
    model.train(False)
    predictions = np.zeros(len(test_dataset))
    targets = np.zeros(len(test_dataset))
    index = 0
    with torch.no_grad():
        for audio, target in tqdm(test_loader):
            audio, target = audio.to(device), target.to(device)
            pred = model(audio).argmax(dim=-1)
            # Aggregate predictions and targets
            cur_batch_size = target.size()[0]
            predictions[index : index + cur_batch_size] = pred.cpu().numpy()
            targets[index : index + cur_batch_size] = target.cpu().numpy()
            index += cur_batch_size
        print("===========================")
        print(f"Test accuracy {accuracy_score(targets, predictions)}!")
        print("===========================")
        _, recalls, _, _ = precision_recall_fscore_support(
            targets, predictions, zero_division=0
        )
        for i, recall in enumerate(recalls):
            print(f"The accuracy for class {i}: {recall}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--checkpoint_path", default="models/best.pt", type=str)
    parser.add_argument("--sampling_rate", default=None, type=int)
    parser.add_argument("--dataset_name", default="data_50", type=str)
    parser.add_argument("--dft_window_size", default=512, type=int)
    parser.add_argument("--hop_length", default=512, type=int)
    parser.add_argument("--mfcc", action="store_true")
    parser.add_argument("--log_mel", action="store_true")
    parser.add_argument("--delta_log_mel", action="store_true")
    args = parser.parse_args()
    inference(
        args.batch_size,
        args.checkpoint_path,
        args.sampling_rate,
        args.dataset_name,
        args.dft_window_size,
        args.hop_length,
    )
