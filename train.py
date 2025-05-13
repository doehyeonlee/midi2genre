#!/usr/bin/env python3
"""
Train and evaluate a multi-label genre classification model on piano-roll numpy arrays using MuSeReNet architecture and settings from config.yaml.
Enhanced to support shallow/deep block types and data augmentation (random transposition) per paper.
"""
import os
import yaml
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import logging
from sklearn.metrics import f1_score, accuracy_score

# ------------------ Configuration ------------------
def load_config(path: str = "config.yaml") -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ------------------ Dataset ------------------
class PianoRollDataset(Dataset):
    def __init__(self, csv_path: str, midi_column: str, label_column: str, data_dir: str, augment: bool = False):
        df = pd.read_csv(csv_path)
        labels_list = df[label_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        unique_labels = sorted({lbl for sub in labels_list for lbl in sub})
        self.class_to_idx = {c: i for i, c in enumerate(unique_labels)}
        self.classes = unique_labels
        self.y = np.zeros((len(df), len(self.classes)), dtype=np.float32)
        for i, sub in enumerate(labels_list):
            for lbl in sub:
                self.y[i, self.class_to_idx[lbl]] = 1.0
        self.filenames = df[midi_column].astype(str).tolist()
        self.data_dir = data_dir
        self.augment = augment

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        base = self.filenames[idx]
        arr = np.load(os.path.join(self.data_dir, f"Midicaps_chunk1_{idx}.npy"))
        if self.augment:
            shift = np.random.randint(-6, 7)
            arr = np.roll(arr, shift, axis=0)
            if shift > 0:
                arr[:shift, :] = 0
            elif shift < 0:
                arr[shift:, :] = 0
        arr = arr.astype(np.float32) / 127.0
        arr = np.expand_dims(arr, 0)
        labels = torch.from_numpy(self.y[idx])
        return torch.from_numpy(arr), labels

# ------------------ MuSeReNet Model ------------------
class MuSeReNet(nn.Module):
    def __init__(self, num_classes: int, block_type: str = 'shallow'):
        super().__init__()
        def conv_block(in_ch):
            layers = []
            if block_type == 'shallow':
                layers += [nn.Conv2d(in_ch, 128, kernel_size=24, padding=12),
                           nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((1,2))]
            else:
                for _ in range(3):
                    layers += [nn.Conv2d(in_ch, 128, kernel_size=9, padding=4),
                               nn.BatchNorm2d(128), nn.ReLU()]
                    in_ch = 128
                layers += [nn.MaxPool2d((1,2))]
            return nn.Sequential(*layers)

        self.block_full = conv_block(1)
        self.block_half = conv_block(1)
        self.block_quarter = conv_block(1)
        self.classifier = nn.Linear(128 * 3, num_classes)

    def forward(self, x):
        out_full = self.block_full(x)
        x_half = F.avg_pool2d(x, kernel_size=(1,2))
        out_half = self.block_half(x_half)
        x_quarter = F.avg_pool2d(x, kernel_size=(1,4))
        out_quarter = self.block_quarter(x_quarter)
        feat_full = F.adaptive_avg_pool2d(out_full, (1,1)).view(x.size(0), -1)
        feat_half = F.adaptive_avg_pool2d(out_half, (1,1)).view(x.size(0), -1)
        feat_quarter = F.adaptive_avg_pool2d(out_quarter, (1,1)).view(x.size(0), -1)
        feats = torch.cat([feat_full, feat_half, feat_quarter], dim=1)
        return self.classifier(feats)

# ------------------ Training & Evaluation ------------------
def train_epoch(model, loader, criterion, optimizer, device, threshold=0.5):
    model.train()
    total_loss, all_preds, all_targets = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) > threshold).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    return avg_loss, f1_score(targets, preds, average='micro', zero_division=0)


def eval_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * x.size(0)
            preds = (torch.sigmoid(logits) > threshold).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    return avg_loss, f1_score(targets, preds, average='micro', zero_division=0)

# ------------------ Training Function ------------------
def train(cfg):
    # Load dataset
    ds = PianoRollDataset(
        cfg['train_csv_chunk'], cfg['csv_file_column'], cfg['label_column'], cfg['output_dir'], augment=cfg.get('augment', False)
    )
    n = len(ds)
    train_n = int(n * cfg.get('train_split', 0.8))
    val_n = n - train_n
    train_set, val_set = random_split(ds, [train_n, val_n])
    train_loader = DataLoader(train_set, batch_size=cfg.get('batch_size',32), shuffle=True, num_workers=cfg.get('num_workers',4))
    val_loader = DataLoader(val_set, batch_size=cfg.get('batch_size',32), shuffle=False, num_workers=cfg.get('num_workers',4))

    # Init model
    device = torch.device(cfg.get('device','cuda'))
    model = MuSeReNet(len(ds.classes), block_type=cfg.get('block_type','shallow')).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.get('lr',1e-3))
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    best_f1 = 0.0
    for epoch in range(1, cfg.get('epochs',50)+1):
        tr_loss, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = eval_epoch(model, val_loader, criterion, device)
        logging.info(f"Epoch {epoch}: Train L={tr_loss:.4f}, F1={tr_f1:.4f} | Val L={val_loss:.4f}, F1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), cfg.get('best_model_path','best_model.pth'))
    logging.info(f"Best validation F1: {best_f1:.4f}")

# ------------------ Main ------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    cfg = load_config(args.config)
    logging.basicConfig(level=getattr(logging, cfg.get('logging_level','INFO').upper()), format='%(asctime)s %(levelname)s %(message)s')
    train(cfg)