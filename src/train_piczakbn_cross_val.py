import argparse
import logging

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import PiczakBNDataset, target2name
from modeling import model_factory
from schedule import get_linear_schedule_with_warmup


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
        "n_features": args.n_features,
        "log_mel": args.log_mel,
        "delta_log_mel": args.delta_log_mel,
        "mfcc": args.mfcc,
        "gfcc": args.gfcc,
        "chroma_stft": args.chroma_stft,
        "cqt": args.cqt,
    }
    data_splits = [
        (["1", "2", "3", "4"], ["5"]),
        (["1", "2", "3", "5"], ["4"]),
        (["1", "2", "4", "5"], ["3"]),
        (["1", "3", "4", "5"], ["2"]),
        (["2", "3", "4", "5"], ["1"]),
    ]
    results = []
    logging.warning(f"Using augmentations: {args.augmentations}")
    for split_num, split in enumerate(data_splits):
        logging.info(f"----------- Starting split number {split_num + 1} -----------")
        # Construct datasets
        train_dataset = PiczakBNDataset(
            args.dataset_path,
            split[0],
            train=True,
            arguments=arguments,
            augmentations=args.augmentations.split(","),
        )
        val_dataset = PiczakBNDataset(
            args.dataset_path,
            split[1],
            train=False,
            arguments=arguments,
            augmentations=[],
        )
        # Construct loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True if args.num_workers else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size // 10,
            num_workers=args.num_workers,
            pin_memory=True if args.num_workers else False,
        )
        # Obtain features
        n_feature_types = np.sum(
            [
                args.log_mel,
                args.delta_log_mel,
                args.mfcc,
                args.gfcc,
                args.chroma_stft,
                args.cqt,
            ]
        )
        # Create model, loss, optimizer, scheduler
        model = model_factory(
            model_type=args.model_type, n_feature_types=n_feature_types
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.epochs,
        )
        logging.info(
            f"Train on {len(train_dataset)}, validate on {len(val_dataset)} samples."
        )
        # Start training
        for epoch in tqdm(range(args.epochs)):
            # Set model in train mode
            model.train(True)
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

            scheduler.step()

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
                audio = audio.view(
                    val_batch_size * 10, n_feature_types, args.n_features, 41
                )
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

            results.append([target2correct, target2total])
            cur_accuracy = sum(target2correct.values()) / sum(target2total.values())
            logging.info("===========================")
            logging.info(
                f"For split number {split_num + 1} the accuracy is {cur_accuracy}"
            )

    # per cl
    class_name2accuracy = {}
    total_accuracy = 0
    for result in results:
        target2correct, target2total = result
        total_accuracy += sum(target2correct.values()) / sum(target2total.values())
        for target in target2total.keys():
            # Obtain class name and class accuracy
            class_name = target2name[target]
            class_accuracy = target2correct[target] / target2total[target]
            if class_name not in class_name2accuracy:
                class_name2accuracy[class_name] = 0
            class_name2accuracy[class_name] += class_accuracy

    logging.info("Printing per-class accuracies:")
    for class_name, class_accuracy in class_name2accuracy.items():
        logging.info(f"{class_name}: {class_accuracy / len(data_splits)}")

    logging.info(f"The average accuracy is {total_accuracy / len(data_splits)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-val LogMel CNN.")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--weight_decay", default=0.0005, type=float)
    parser.add_argument("--dataset_path", default="data/data_50_numpy_22050/", type=str)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--log_mel", type=bool, default=False)
    parser.add_argument("--delta_log_mel", type=bool, default=False)
    parser.add_argument("--mfcc", type=bool, default=False)
    parser.add_argument("--gfcc", type=bool, default=False)
    parser.add_argument("--chroma_stft", type=bool, default=False)
    parser.add_argument("--cqt", type=bool, default=False)
    parser.add_argument("--n_features", default=60, type=int)
    parser.add_argument("--dft_window_size", default=1024, type=int)
    parser.add_argument("--model_type", default="batch_norm", type=str)
    parser.add_argument("--hop_length", default=512, type=int)
    parser.add_argument("--log_filepath", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--augmentations",
        default="",
        type=str,
        help="Comma-separated augmentations: pitch_shift,gaussian_noise",
    )
    args = parser.parse_args()
    train_model(args)
