#!/usr/bin/env python3
"""
Train and evaluate a multi-label genre classification model on piano-roll numpy arrays using settings from config.yaml.
"""
import os
import yaml
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score
import logging
import argparse

# ------------------ Configuration ------------------
def load_config(path: str = "config.yaml") -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ------------------ Dataset ------------------
class PianoRollDataset(Dataset):
    def __init__(self, csv_path: str, midi_column: str, label_column: str, data_dir: str):
        df = pd.read_csv(csv_path)
        labels_list = df[label_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        unique_labels = sorted({lbl for sub in labels_list for lbl in sub})
        self.class_to_idx = {c: i for i, c in enumerate(unique_labels)}
        self.classes = unique_labels

        self.y = np.zeros((len(df), len(self.classes)), dtype=np.float32)
        for i, sub in enumerate(labels_list):
            for lbl in sub:
                idx = self.class_to_idx.get(lbl)
                if idx is not None:
                    self.y[i, idx] = 1.0

        self.filenames = df[midi_column].astype(str).tolist()
        self.data_dir = data_dir

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        base = self.filenames[idx]
        arr = np.load(os.path.join(self.data_dir, f"{idx}.npy"))
        arr = arr.astype(np.float32) / 127.0
        arr = np.expand_dims(arr, 0)  # (1, 128, frames)
        labels = torch.from_numpy(self.y[idx])
        return torch.from_numpy(arr), labels

# ------------------ Model ------------------
class GenreCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ------------------ Training & Evaluation ------------------
def train_epoch(model, loader, criterion, optimizer, device, threshold=0.5):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float().cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    f1 = f1_score(targets, preds, average='micro', zero_division=0)
    return avg_loss, f1

def eval_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    f1 = f1_score(targets, preds, average='micro', zero_division=0)
    return avg_loss, f1

# ------------------ Main ------------------
def train(cfg):
    level = getattr(logging, cfg.get('logging_level', 'INFO').upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(message)s')

    dataset = PianoRollDataset(
        cfg['train_csv_chunk'],
        cfg['csv_file_column'],
        cfg['label_column'],
        cfg['output_dir']
    )

    n = len(dataset)
    train_n = int(n * cfg.get('train_split', 0.8))
    val_n = n - train_n
    train_set, val_set = random_split(dataset, [train_n, val_n])
    train_loader = DataLoader(train_set, batch_size=cfg.get('batch_size', 32), shuffle=True, num_workers=cfg.get('num_workers', 4))
    val_loader = DataLoader(val_set, batch_size=cfg.get('batch_size', 32), shuffle=False, num_workers=cfg.get('num_workers', 4))

    model = GenreCNN(len(dataset.classes)).to(torch.device(cfg.get('device', 'cuda')))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.get('lr', 1e-3))

    best_f1 = 0.0
    for epoch in range(1, cfg.get('epochs', 50) + 1):
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, cfg['device'])
        val_loss, val_f1 = eval_epoch(model, val_loader, criterion, cfg['device'])
        logging.info(f"Epoch {epoch}: Train loss={train_loss:.4f}, f1={train_f1:.4f} | Val loss={val_loss:.4f}, f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), cfg.get('best_model_path', 'best_model.pth'))

    logging.info(f"Best validation F1: {best_f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
    