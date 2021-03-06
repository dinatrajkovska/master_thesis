import os
import argparse
import logging

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import EnvNetDataset, target2name
from modeling import envnet_v2
from schedule import get_linear_schedule_with_warmup


def train_model(args):
    # Set up logging
    if args.log_filepath:
        logging.basicConfig(
            level=logging.INFO, filename=args.log_filepath, filemode="w"
        )
    else:
        logging.basicConfig(level=logging.INFO)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        for i in range(torch.cuda.device_count()):
            logging.warning(f"{torch.cuda.get_device_properties(f'cuda:{i}')}")
            logging.warning(
                f"Current occupied memory: {torch.cuda.memory_allocated(i) * 1e-9} GB"
            )
    else:
        device = torch.device("cpu")
    logging.warning(f"Using device {device}!!!")
    # Prepare datasets
    folds = args.folds.split(",")
    logging.info(f"Training folds: {folds[:4]}")
    train_dataset = EnvNetDataset(
        args.dataset_path,
        folds[:4],
        train=True,
        augmentations=args.augmentations.split(","),
    )
    logging.info(f"Testing fold: {folds[4:]}")
    val_dataset = EnvNetDataset(
        args.dataset_path, folds[4:], train=False, augmentations=[]
    )
    # Prepare dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size // 10)
    model = envnet_v2().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs
    )
    logging.info(
        f"Train on {len(train_dataset)}, validate on {len(val_dataset)} samples."
    )
    cur_epoch = 0
    if os.path.exists(args.checkpoint_path):
        # https://discuss.pytorch.org/t/saving-model-and-optimiser-and-scheduler/52030/8
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        cur_epoch = checkpoint["epoch"]
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # https://discuss.pytorch.org/t/cuda-out-of-memory-after-loading-model/50681
        del checkpoint
        logging.warning(
            f"Starting training from checkpoint {args.checkpoint_path} with starting epoch {cur_epoch+1}!"
        )

    for epoch in range(cur_epoch, args.epochs):
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

        scheduler.step()

        # Save training state dict
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            args.checkpoint_path,
        )

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

    accuracy = sum(target2correct.values()) / sum(target2total.values())
    logging.info("===========================")
    logging.info(f"The accuracy is {accuracy}!")
    logging.info("Per-class accuracies:")
    for target in target2total.keys():
        logging.info(
            f"{target2name[target]}: {target2correct[target]/target2total[target]}"
        )
    logging.info("===========================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EnvNet on holdout.")
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=0.002, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--dataset_path", default="data/data_50_numpy_22050", type=str)
    parser.add_argument("--log_filepath", type=str, default=None)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--folds", default="1,2,3,4,5", type=str)
    parser.add_argument(
        "--augmentations",
        default="none",
        type=str,
        help="Comma-separated augmentations: pitch_shift,gaussian_noise",
    )
    args = parser.parse_args()
    train_model(args)
