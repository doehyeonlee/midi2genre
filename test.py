#!/usr/bin/env python3
"""
Evaluate a trained MuSeReNet genre classification model imported from train.py on a test set.
Usage:
    python test.py --config config.yaml
"""
import os
import yaml
import ast
import numpy as np
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from train import MuSeReNet  # import model architecture
import pandas as pd
# ------------------ Configuration ------------------
def load_config(path: str = "config.yaml") -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ------------------ Dataset ------------------
class TestPianoRollDataset(Dataset):
    def __init__(self, csv_path, midi_column, label_column, data_dir):
        df = __import__('pandas').read_csv(csv_path)
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

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        arr = np.load(os.path.join(self.data_dir, f"{idx}.npy"))
        arr = arr.astype(np.float32) / 127.0
        arr = np.expand_dims(arr, 0)
        labels = torch.from_numpy(self.y[idx])
        return torch.from_numpy(arr), labels

# ------------------ Test Function ------------------
def test(cfg):
    # dataset & loader
    ds = TestPianoRollDataset(
        cfg['test_csv_chunk'],
        cfg['csv_file_column'],
        cfg['label_column'],
        cfg['test_dataset_dir']
    )
    loader = DataLoader(ds, batch_size=cfg.get('batch_size',32), shuffle=False, num_workers=cfg.get('num_workers',4))

    # model
    device = torch.device(cfg.get('device','cuda'))
    model = MuSeReNet(len(ds.classes), block_type=cfg.get('block_type','shallow')).to(device)
    model.load_state_dict(torch.load(cfg.get('best_model_path','best_model.pth'), map_location=device))
    model.eval()

    # inference
    all_targets, all_preds = [], []
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > cfg.get('threshold',0.5)).astype(int)
            all_preds.append(preds)
            all_targets.append(y.cpu().numpy())
    targets = np.vstack(all_targets)
    preds = np.vstack(all_preds)

    # metrics
    micro_f1 = f1_score(targets, preds, average='micro', zero_division=0)
    acc = accuracy_score(targets, preds)
    print(f"Test Accuracy: {acc:.4f}, Micro-F1: {micro_f1:.4f}")

    # per-class metrics

    records = []
    for i, cls in enumerate(ds.classes):
        cls_true = targets[:,i]
        cls_pred = preds[:,i]
        records.append({
            'genre': cls,
            'accuracy': accuracy_score(cls_true, cls_pred),
            'f1': f1_score(cls_true, cls_pred, zero_division=0)
        })
    df = pd.DataFrame(records)
    path = cfg.get('result_csv_path','test_results.csv')
    df.to_csv(path, index=False)
    print(f"Per-genre metrics saved to {path}")

# ------------------ Main ------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    cfg = load_config(args.config)
    logging.basicConfig(level=getattr(logging, cfg.get('logging_level','INFO').upper()), format='%(asctime)s %(levelname)s %(message)s')
    test(cfg)
