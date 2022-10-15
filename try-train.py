import enum
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab

from torch.utils.data import DataLoader
from model import SeqClassifier
import torch.optim as optim
import os
import csv

TRAIN = "train"
DEV = "eval"
TEST = "test"
SPLITS = [TRAIN, DEV, TEST]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # datasets = {TRAIN: SeqClsDataset, DEV: SeqClsDataset}
    # TODO: crecate DataLoader for train / dev datasets
    train_dataloader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, collate_fn=datasets["train"].collate_fn)

    # train_dataloader = DataLoader(datasets["train"], batch_size=2, shuffle=True)
    # x, y = next(iter(train_dataloader))
    # x = list(x) # tuple to list
    # x = vocab.encode_batch([sentence.split() for sentence in x])
    
    eval_dataloader = DataLoader(datasets["eval"], batch_size=args.batch_size, shuffle=False, collate_fn=datasets["eval"].collate_fn)
    test_dataloader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, collate_fn=datasets["test"].collate_fn)
    
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    
    # x: [batch size, length, H_in]

    # TODO: init model and move model to target device(cpu / gpu)
    num_class = datasets["train"].num_classes
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, num_class=num_class)
    model = model.to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()


    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        best_eval_acc = 0
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        loss_train = 0
        acc_train = 0
        for step, (x, y, _) in enumerate(train_dataloader):

            x = torch.tensor(x).to(args.device)
            y = torch.tensor(y).to(args.device)
            
            output = model(x)            
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loss_train += loss.item()
            output = torch.argmax(output, dim=1).to(float)
            acc_train += torch.eq(output, y).sum().float().item()

        loss_train /= (step + 1)
        acc_train /= (len(datasets["train"]))
        print(f"loss_train: {loss_train}; acc_train: {acc_train}")

        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        with torch.no_grad():
            loss_eval = 0
            acc_eval = 0
            for step, (x, y, _) in enumerate(eval_dataloader):

                x = torch.tensor(x).to(args.device)                
                y = torch.tensor(y).to(args.device)
                
                output = model(x)
                
                loss = criterion(output, y)
                
                loss_eval += loss.item()
                output = torch.argmax(output, dim=1).to(float)
                acc_eval += torch.eq(output, y).sum().float().item()
            loss_eval /= (step + 1)
            acc_eval /= (len(datasets["eval"]))
            print(f"loss_eval: {loss_eval}; acc_eval: {acc_eval}")

		# save model
        if acc_eval >= best_eval_acc:
            best_eval_acc = acc_eval
            
            torch.save(model.state_dict(), 
                        os.path.join(args.ckpt_dir, 
                                    f"{args.hidden_size}-{args.num_layers}-\
                                    {args.dropout}-{args.bidirectional}- \
                                    {args.lr}-{args.batch_size}-{args.num_epoch}-model.pt"))
    # TODO: Inference on test set

	# Inference on test set
	# first load-in best model

    # model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "model.pt")))
    
    model.eval()
    with torch.no_grad():
        ansList = [['id', 'intent']]

        for step, (x, id) in enumerate(test_dataloader):
            
            x = torch.tensor(x).to(args.device)
            
            output = model(x)
            
            output = torch.argmax(output, dim=1)

            for (i, label) in zip(id, output):
                ansList.append([i, datasets['test'].idx2label(label.item())])
    fileanme = f"{args.hidden_size}-{args.num_layers}-\
                                    {args.dropout}-{args.bidirectional}- \
                                    {args.lr}-{args.batch_size}-{args.num_epoch}-intent-ans.csv"
    with open(fileanme, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(ansList)
                


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=5e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=512)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
