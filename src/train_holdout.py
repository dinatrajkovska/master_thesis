import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse
import logging
from typing import Dict

from datasets import AudioDataset, target2name
from modeling import pitzak_factory


def major_vote(mini_batch: torch.Tensor) -> int:
    max_value_dict: Dict[int, int] = {}
    for elem in mini_batch:
        item = elem.item()
        if item not in max_value_dict:
            max_value_dict[item] = 0
        max_value_dict[item] += 1
    max_val = 0
    max_key = ""
    for key, val in max_value_dict.items():
        if val > max_val:
            max_val = val
            max_key = key
    return max_key


def train_model(args):
    # Set up logging
    if args.log_filepath:
        logging.basicConfig(
            level=logging.INFO, filename=args.log_filepath, filemode="w"
        )
    else:
        logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"-------- Using device {device} --------")

    arguments = {
        "n_fft": args.dft_window_size,
        "hop_length": args.hop_length,
        "num_mels": args.n_mels,
    }
    logging.info("==================================")
    logging.info("Features used: ")
    logging.info(f"Log mel spectogram: {args.log_mel}")
    logging.info(f"Delta log mel spectogram: {args.delta_log_mel}")
    logging.info(f"Mel-frequency cepstral coefficients: {args.mfcc}")
    logging.info(f"Gammatone-frequency cepstral coefficients: {args.gfcc}")
    logging.info(f"Constant-Q transform: {args.cqt}")
    logging.info(f"STFT chromagram: {args.chroma}")
    logging.info("==================================")
    train_dataset = AudioDataset(
        args.dataset_path,
        args.gammatones_path,
        [1, 2, 3, 4],
        arguments,
        args.sampling_rate,
        args.log_mel,
        args.delta_log_mel,
        args.mfcc,
        args.gfcc,
        args.cqt,
        args.chroma,
    )

    val_dataset = AudioDataset(
        args.dataset_path,
        args.gammatones_path,
        [5],
        arguments,
        args.sampling_rate,
        args.log_mel,
        args.delta_log_mel,
        args.mfcc,
        args.gfcc,
        args.cqt,
        args.chroma,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    in_features = np.sum(
        [args.log_mel, args.delta_log_mel, args.mfcc, args.gfcc, args.cqt, args.chroma]
    )
    assert in_features > 0
    model = pitzak_factory(args.model_type, in_features).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    logging.info(
        f"Train on {len(train_dataset)}, validate on {len(val_dataset)} samples."
    )
    best_accuracy = -1
    best_epoch = -1
    for epoch in range(args.epochs):
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
        predictions = np.zeros(len(val_dataset) // 9)
        targets = np.zeros(len(val_dataset) // 9)
        target2total = {}
        target2correct = {}
        index = 0
        with torch.no_grad():
            for _, audio, target in tqdm(val_loader):
                audio, target = audio.to(device), target.to(device)
                prediction = model(audio).argmax(-1)
                t_prediction = torch.zeros(prediction.size()[0] // 9)
                for count, i in enumerate(range(0, prediction.size()[0], 9)):
                    t_prediction[count] = major_vote(prediction[i : i + 9])
                t_target = target[::9]
                cur_batch_size = t_target.size()[0]
                # Aggregate predictions and targets
                predictions[index : index + cur_batch_size] = t_prediction.cpu().numpy()
                targets[index : index + cur_batch_size] = t_target.cpu().numpy()
                index += cur_batch_size
            print(predictions, targets)
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
                torch.save(model.state_dict(), args.save_model_path)
            else:
                logging.info(f"Epoch {epoch+1} with accuracy {cur_accuracy}!")

    logging.info("===========================")
    logging.info(f"Best total accuracy {best_accuracy} on epoch {best_epoch}")
    logging.info("===========================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on holdout.")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=0.002, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--n_mels", default=60, type=int)
    parser.add_argument("--save_model_path", default="models/best.pt", type=str)
    parser.add_argument("--mfcc", action="store_true")
    parser.add_argument("--log_mel", action="store_true")
    parser.add_argument("--delta_log_mel", action="store_true")
    parser.add_argument("--cqt", action="store_true")
    parser.add_argument("--chroma", action="store_true")
    parser.add_argument("--gfcc", action="store_true")
    parser.add_argument("--dataset_path", default="data/data_50/", type=str)
    parser.add_argument("--model_type", default="regular", type=str)
    parser.add_argument(
        "--gammatones_path", default="data/gammatone_features", type=str
    )
    parser.add_argument("--dft_window_size", default=1024, type=int)
    parser.add_argument("--hop_length", default=512, type=int)
    parser.add_argument("--log_filepath", type=str, default=None)
    parser.add_argument("--sampling_rate", type=int, default=None)
    args = parser.parse_args()
    train_model(args)

