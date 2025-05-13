#!/usr/bin/env python3
"""
Evaluate a trained MuSeReNet single-label genre classification model on piano-roll numpy arrays.
- Uses only the first genre label from CSV
- Predicts single class (argmax)
- Computes and saves per-genre Accuracy, F1-score, Balanced Accuracy, Support
- Also reports macro-averaged metrics
"""
import os
import yaml
import ast
import numpy as np
import torch
import logging
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from train import MuSeReNet

# ------------------ Configuration ------------------
def load_config(path: str = "config.yaml") -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ------------------ Dataset ------------------
class TestPianoRollDataset(Dataset):
    def __init__(self, csv_path, file_column, label_column, data_dir):
        df = pd.read_csv(csv_path)
        # use only first genre label
        labels = df[label_column].apply(
            lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x
        )
        self.classes = sorted(labels.unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.targets = labels.map(self.class_to_idx).values
        self.filenames = df[file_column].astype(str).tolist()
        self.data_dir = data_dir

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        arr = np.load(
            os.path.join(self.data_dir, f"Midicaps_chunk2_{idx}.npy")
        )  # shape (128, T)
        arr = arr.astype(np.float32) / 127.0
        arr = np.expand_dims(arr, 0)  # (1,128,T)
        label = self.targets[idx]
        return torch.from_numpy(arr), label

# ------------------ Test Logic ------------------
def test(cfg):
    # load dataset
    ds = TestPianoRollDataset(
        cfg['test_csv_chunk'],
        cfg['csv_file_column'],
        cfg['label_column'],
        cfg['test_dataset_dir']
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.get('batch_size', 32),
        shuffle=False,
        num_workers=cfg.get('num_workers', 4)
    )

    # load model
    device = torch.device(cfg.get('device', 'cpu'))
    model = MuSeReNet(len(ds.classes), block_type=cfg.get('block_type', 'shallow'))
    model.load_state_dict(
        torch.load(cfg.get('best_model_path', 'best_model.pth'), map_location=device)
    )
    model.to(device).eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y)

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # compute per-class metrics
    records = []
    acc_list, f1_list, bacc_list, support_list = [], [], [], []
    for i, cls in enumerate(ds.classes):
        true = (targets == i).astype(int)
        pred = (preds == i).astype(int)
        support = int(true.sum())
        acc = accuracy_score(true, pred)
        f1 = f1_score(true, pred, zero_division=0)
        bacc = balanced_accuracy_score(true, pred)
        records.append({
            'genre': cls,
            'accuracy': acc,
            'f1': f1,
            'balanced_accuracy': bacc,
            'support': support
        })
        acc_list.append(acc)
        f1_list.append(f1)
        bacc_list.append(bacc)
        support_list.append(support)

    # macro metrics
    macro_acc = np.mean(acc_list)
    macro_f1 = np.mean(f1_list)
    macro_bacc = np.mean(bacc_list)
    print(f"Macro-Accuracy: {macro_acc:.4f}, Macro-F1: {macro_f1:.4f}, Macro-Balanced-Accuracy: {macro_bacc:.4f}")

    # save per-genre
    df = pd.DataFrame(records)
    out_csv = cfg.get('result_csv_path', 'test_results.csv')
    df.to_csv(out_csv, index=False)
    print(f"Saved per-genre metrics to {out_csv}")
    print(df)

# ------------------ Main ------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    cfg = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, cfg.get('logging_level', 'INFO').upper()),
        format='%(asctime)s %(levelname)s %(message)s'
    )
    test(cfg)
