#!/usr/bin/env python3
"""
Evaluate a trained MuSeReNet single-label or multi-label genre classification model on piano-roll numpy arrays.
Supports both top-1 and top-k evaluation via `--top_k` argument.
Computes per-genre Accuracy, Precision, Recall, F1-score, Balanced Accuracy, Support,
and macro + overall metrics. Saves results to CSV.
"""
import os
import yaml
import ast
import numpy as np
import torch
import logging
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from train import MuSeReNet

# ------------------ Configuration ------------------
def load_config(path: str = "config.yaml") -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ------------------ Dataset ------------------
class TestPianoRollDataset(Dataset):
    def __init__(self, csv_path, file_column, label_column, data_dir, top_k=1, json_path="label_index.json"):
        import json
        # Load label-index mapping from JSON
        with open(json_path, "r") as f:
            self.class_to_idx = json.load(f)
        # Build classes list in index order
        self.classes = [None] * len(self.class_to_idx)
        for lbl, idx in self.class_to_idx.items():
            self.classes[idx] = lbl
        self.top_k = top_k

        df = pd.read_csv(csv_path)[2000:3000]
        # Parse label lists
        labels_list = df[label_column].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else [x]
        ).tolist()
        # Prepare targets
        if top_k == 1:
            # single-label: first label
            self.targets = np.array([self.class_to_idx[sub[0]] for sub in labels_list])
        else:
            # multi-label: multi-hot up to top_k labels
            self.targets = np.zeros((len(df), len(self.classes)), dtype=int)
            for i, sub in enumerate(labels_list):
                for lbl in sub[:top_k]:
                    if lbl in self.class_to_idx:
                        self.targets[i, self.class_to_idx[lbl]] = 1

        self.filenames = df[file_column].astype(str).tolist()
        self.data_dir = data_dir

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Load precomputed piano-roll numpy
        arr = np.load(
            os.path.join(self.data_dir, f"Midicaps_chunk2_{idx}.npy")
        )  # shape (128, T)
        arr = arr.astype(np.float32) / 127.0
        arr = np.expand_dims(arr, 0)  # (1,128,T)
        label = self.targets[idx]
        return torch.from_numpy(arr), label

# ------------------ Test Logic ------------------
def test(cfg):
    top_k = cfg.get('top_k', 1)
    json_path = cfg.get('json_path', 'label_index.json')
    ds = TestPianoRollDataset(
        cfg['test_csv_chunk'], cfg['csv_file_column'], cfg['label_column'], cfg['test_dataset_dir'], top_k, json_path
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.get('batch_size', 32),
        shuffle=False,
        num_workers=cfg.get('num_workers', 4)
    )

    device = torch.device(cfg.get('device', 'cpu'))
    model = MuSeReNet(len(ds.classes), block_type=cfg.get('block_type', 'shallow'))
    model.load_state_dict(torch.load(cfg.get('best_model_path', 'best_model.pth'), map_location=device))
    model.to(device).eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            if top_k == 1:
                pred = logits.argmax(dim=1).cpu().numpy()
            else:
                probs = torch.sigmoid(logits).cpu().numpy()
                top_inds = np.argsort(probs, axis=1)[:, -top_k:]
                pred = np.zeros_like(probs, dtype=int)
                for i, inds in enumerate(top_inds):
                    pred[i, inds] = 1
            all_preds.append(pred)
            # convert targets
            if isinstance(y, torch.Tensor):
                y_np = y.cpu().numpy()
            else:
                y_np = np.array(y)
            all_targets.append(y_np)

    # concatenate across batches
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # compute per-class metrics
    records = []
    for i, cls in enumerate(ds.classes):
        if top_k == 1:
            true = (targets == i).astype(int)
            pre = (preds == i).astype(int)
        else:
            true = targets[:, i]
            pre = preds[:, i]
        support = int(true.sum())
        acc = accuracy_score(true, pre)
        prec = precision_score(true, pre, zero_division=0)
        rec = recall_score(true, pre, zero_division=0)
        f1 = f1_score(true, pre, zero_division=0)
        bacc = balanced_accuracy_score(true, pre)
        records.append({
            'genre': cls,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'balanced_accuracy': bacc,
            'support': support
        })

    # overall metrics
    if top_k == 1:
        overall_acc = accuracy_score(targets, preds)
        overall_prec = precision_score(targets, preds, average='micro', zero_division=0)
        overall_rec = recall_score(targets, preds, average='micro', zero_division=0)
        overall_f1 = f1_score(targets, preds, average='micro', zero_division=0)
        overall_bacc = balanced_accuracy_score(targets, preds)
    else:
        overall_acc = accuracy_score(targets.flatten(), preds.flatten())
        overall_prec = precision_score(targets.flatten(), preds.flatten(), zero_division=0)
        overall_rec = recall_score(targets.flatten(), preds.flatten(), zero_division=0)
        overall_f1 = f1_score(targets.flatten(), preds.flatten(), zero_division=0)
        overall_bacc = balanced_accuracy_score(targets.flatten(), preds.flatten())
    overall_sup = len(targets) if top_k == 1 else targets.sum()

    print(f"Overall: Acc={overall_acc:.4f}, Prec={overall_prec:.4f}, Rec={overall_rec:.4f}, "
          f"F1={overall_f1:.4f}, BalAcc={overall_bacc:.4f}, Support={overall_sup}")

    df = pd.DataFrame(records)
    df.loc[len(df)] = {
        'genre': 'Overall',
        'accuracy': overall_acc,
        'precision': overall_prec,
        'recall': overall_rec,
        'f1': overall_f1,
        'balanced_accuracy': overall_bacc,
        'support': overall_sup
    }
    out_csv = cfg.get('result_csv_path', 'test_results.csv')
    df.to_csv(out_csv, index=False)
    print(f"Saved metrics to {out_csv}")
    print(df)

# ------------------ Main ------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--top_k', type=int, default=2, help='Number of top labels to predict')
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg['top_k'] = args.top_k
    logging.basicConfig(level=getattr(logging, cfg.get('logging_level','INFO').upper()),
                        format='%(asctime)s %(levelname)s %(message)s')
    test(cfg)
