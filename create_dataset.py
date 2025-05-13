import os
import yaml
import logging
import shutil
import tarfile
import numpy as np
import pretty_midi
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import argparse
from huggingface_hub import hf_hub_download

def setup_logging(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=lvl
    )

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def download_midi_files(cfg: dict):
    os.makedirs(cfg["origin_dir"], exist_ok=True)
    os.makedirs("/data/anthony16/midicaps_train/cache", exist_ok=True)
    logging.info(f"Loading HF dataset {cfg['dataset_name']} split={cfg['dataset_split']}")
    cache_path = hf_hub_download(
        repo_id=cfg["dataset_name"],       
        filename="midicaps.tar.gz",  
        repo_type="dataset",               
        cache_dir=cfg["download_dir"] 
    )
    target_path = os.path.join(cfg["origin_dir"], "midicaps.tar.gz")
    shutil.copy(cache_path, target_path)

    with tarfile.open(target_path, "r:gz") as tar:
        tar.extractall(path=cfg["origin_dir"])
    print(f"✔ Extracted to {cfg["origin_dir"]}")
    

def midi_to_pianoroll(path: str, fs: int, length: int) -> np.ndarray | None:
    try:
        pm = pretty_midi.PrettyMIDI(path)
        roll = pm.get_piano_roll(fs=fs)[:128]    # (128, T)
        roll = (roll > 0).astype(np.float32).T    # (T,128)
        if roll.shape[0] >= length:
            return roll[:length]
        pad = np.zeros((length - roll.shape[0], 128), dtype=np.float32)
        return np.vstack([roll, pad])
    except Exception as e:
        logging.warning(f"Pianoroll failed ({path}): {e}")
        return None
    

def convert_all_midi(cfg: dict):
    os.makedirs(cfg["test_dataset_dir"], exist_ok=True)

    df = pd.read_csv(cfg["test_csv_chunk"])

    for idx, base_name in tqdm(df[cfg["csv_file_column"]].astype(str).items(), desc="Converting to piano-roll"):
        src = os.path.join(cfg["origin_dir"], base_name)
        if not os.path.isfile(src):
            print(f"[Warning] File not found: {src}")
            continue
        out_path = os.path.join(cfg["test_dataset_dir"], f"Midicaps_chunk2_{idx}.npy")
        roll = midi_to_pianoroll(src, cfg["fs"], cfg["target_length"])
        if roll is not None:
            np.save(out_path, roll)
        
def convert_all_midi_customdir(cfg: dict):
    os.makedirs(cfg["custom_dataset_dir"], exist_ok=True)

    df = pd.read_csv(cfg["custom_csv_chunk"])

    for idx, _ in tqdm(df[cfg["csv_file_column"]].astype(str).items(), desc="Converting to piano-roll"):
        padded_idx = str(idx).zfill(5)
        src = os.path.join(cfg["custom_origin_dir"], f"{padded_idx}.mid")
        if not os.path.isfile(src):
            print(f"[Warning] File not found: {src}")
            continue
        out_path = os.path.join(cfg["custom_dataset_dir"], f"Midicaps_{cfg['custom_method_name']}_{idx}.npy")
        roll = midi_to_pianoroll(src, cfg["fs"], cfg["target_length"])
        if roll is not None:
            np.save(out_path, roll)

def main():
    parser = argparse.ArgumentParser(
        description="1) Download selected MIDI  2) Convert to piano-roll"
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging_level", "INFO"))
    logging.info("=== Pipeline started ===")

    #download_midi_files(cfg)
    #convert_all_midi(cfg)
    convert_all_midi_customdir(cfg)
    logging.info("=== Pipeline finished ===")

if __name__ == "__main__":
    main()