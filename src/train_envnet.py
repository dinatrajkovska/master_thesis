import argparse
import logging

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import EnvNetDataset, target2name


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

    # Prepare datasets
    train_dataset = EnvNetDataset(args.dataset_path, [1, 2, 3, 4], train=True)
    val_dataset = EnvNetDataset(args.dataset_path, [5], train=False)
    # Prepare dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size // 10)
    model = 5
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
            for audio, target in train_loader:
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
        target2total = {}
        target2correct = {}
        predictions = np.zeros(len(val_dataset))
        targets = np.zeros(len(val_dataset))
        index = 0
        with torch.no_grad():
            for audio, target in tqdm(val_loader):
                audio, target = audio.to(device), target.to(device)
                # Obtain dimensions
                val_batch_size = audio.size()[0]
                num_samples = audio.size()[-1]
                audio = audio.view(val_batch_size * 10, 1, 1, num_samples)
                # Obtain predictions: [Val-BS * 10, Num-classes]
                probs = model(audio)
                num_classes = probs.size()[-1]
                # [Val-BS, 10, 50]
                prediction = probs.view(val_batch_size, 10, num_classes)
                # [Val-BS, 50]
                prediction = prediction.mean(1)
                # [Val-BS, 1]
                prediction = prediction.argmax(-1)
                # Aggregate predictions and targets
                predictions[index : index + val_batch_size] = prediction.cpu().numpy()
                targets[index : index + val_batch_size] = target.cpu().numpy()
                index += val_batch_size

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
                    f"Best on epoch {best_epoch} with accuracy {best_accuracy}! Saving..."
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
    parser = argparse.ArgumentParser(description="Train EnvNet on holdout.")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=0.002, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--save_model_path", default="models/best.pt", type=str)
    parser.add_argument("--dataset_path", default="data_50", type=str)
    parser.add_argument("--log_filepath", type=str, default=None)
    args = parser.parse_args()
    train_model(args)
