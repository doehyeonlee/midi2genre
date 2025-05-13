import os
import yaml
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def setup_logging(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=lvl
    )

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_genre_mapping(df: pd.DataFrame, genre_col: str, delimiter: str) -> tuple[dict, int]:
    # CSV 전체를 보고 가능한 장르 집합을 수집
    genres = sorted({g for entry in df[genre_col] for g in entry.split(delimiter)})
    genre2idx = {g: i for i, g in enumerate(genres)}
    return genre2idx, len(genres)

class GenreDataset(Dataset):
    def __init__(self, config: dict):
        self.cfg = config

        # CSV 로드
        df = pd.read_csv(self.cfg["csv_path"])
        self.genre2idx, self.num_classes = build_genre_mapping(
            df,
            genre_col=self.cfg["genre_column"],
            delimiter=self.cfg["split_delimiter"]
        )

        self.samples = []
        missing = 0
        for _, row in df.iterrows():
            # 파일 이름 → .npy 경로
            fname = row[self.cfg["file_column"]]
            fname = os.path.splitext(fname)[0] + ".npy"
            path = os.path.join(self.cfg["npy_dir"], fname)

            if not os.path.isfile(path):
                missing += 1
                if not self.cfg["ignore_missing"]:
                    raise FileNotFoundError(f"Missing file: {path}")
                continue

            # one-hot 레이블 생성
            labels = row[self.cfg["genre_column"]].split(self.cfg["split_delimiter"])
            onehot = np.zeros(self.num_classes, dtype=np.float32)
            for g in labels:
                idx = self.genre2idx.get(g)
                if idx is not None:
                    onehot[idx] = 1.0

            self.samples.append((path, onehot))

        logging.info(f"Found {len(self.samples)} samples, skipped {missing} missing files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = np.load(path)
        return torch.from_numpy(x), torch.from_numpy(label)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GenreDataset 확인")
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="YAML 설정 파일 경로"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging_level", "INFO"))
    logging.info("Dataset 설정 로드 완료")
    ds = GenreDataset(cfg)
    logging.info(f"Dataset 크기: {len(ds)}")
    # 샘플 하나 출력해보기
    x, y = ds[0]
    logging.info(f"첫 샘플: x.shape={x.shape}, y.sum()={y.sum()}")
