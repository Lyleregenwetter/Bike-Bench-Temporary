import os
import json
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from biked_commons.resource_utils import datasets_path  # renamed from split_datasets_path

# Load Dataverse file ID map
with open(datasets_path("dataverse_file_ids.json")) as f:
    FILE_ID_MAP = json.load(f)

BASE_URL = "https://dataverse.harvard.edu/api/access/datafile"


def download_file(file_id: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    url = f"{BASE_URL}/{file_id}"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        with open(dest_path, "wb") as f_out, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {os.path.basename(dest_path)}"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f_out.write(chunk)
                    pbar.update(len(chunk))
        print(f"✅ Downloaded {os.path.basename(dest_path)} to {dest_path}")
    else:
        raise RuntimeError(f"Failed to download file: {file_id} (HTTP {response.status_code})")


def download_if_missing(remote_path: str):
    local_path = datasets_path(remote_path)
    if not os.path.exists(local_path):
        print(f"⚠️  {os.path.basename(local_path)} not found locally. Downloading from Harvard Dataverse...")
        file_id = FILE_ID_MAP.get(remote_path)
        if not file_id:
            raise ValueError(f"File ID not found for: {remote_path}")
        download_file(file_id, local_path)
    return local_path


def load_any_file(filepath: str):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".tab":
        return pd.read_csv(filepath, index_col=0, sep="\t")
    elif ext == ".csv":
        return pd.read_csv(filepath, index_col=0)
    elif ext == ".npy":
        return np.load(filepath, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")


def load_dataset_pair(name_prefix: str, folder: str, y_ext: str = None, y_train_is_npy: bool = False):
    # Load train
    x_train_path = download_if_missing(f"{folder}/{name_prefix}_X_train.csv")
    y_train_name = f"{name_prefix}_Y_train.npy" if y_train_is_npy else f"{name_prefix}_Y_train.{y_ext or 'csv'}"
    y_train_path = download_if_missing(f"{folder}/{y_train_name}")

    X_train = load_any_file(x_train_path)
    Y_train = load_any_file(y_train_path)

    return X_train, Y_train


def load_dataset_pair_test(name_prefix: str, folder: str, y_ext: str = None):
    x_test_path = download_if_missing(f"{folder}/{name_prefix}_X_test.csv")
    y_test_name = f"{name_prefix}_Y_test.{y_ext or 'csv'}"
    y_test_path = download_if_missing(f"{folder}/{y_test_name}")

    X_test = load_any_file(x_test_path)
    Y_test = load_any_file(y_test_path)

    return X_test, Y_test

# ---- Predictive modeling dataset functions ----

# Aero
def load_aero_train(): return load_dataset_pair("aero", "Predictive_Modeling_Datasets")
def load_aero_test(): return load_dataset_pair_test("aero", "Predictive_Modeling_Datasets")

# Structure
def load_structure_train(): return load_dataset_pair("structure", "Predictive_Modeling_Datasets")
def load_structure_test(): return load_dataset_pair_test("structure", "Predictive_Modeling_Datasets")

def one_hot_encode_material(data: pd.DataFrame):
    data = data.copy()
    # One-hot encode the materials
    data["Material"] = pd.Categorical(data["Material"], categories=["Steel", "Aluminum", "Titanium"])
    mats_oh = pd.get_dummies(data["Material"], prefix="Material=", prefix_sep="")
    data.drop(["Material"], axis=1, inplace=True)
    data = pd.concat([mats_oh, data], axis=1)
    return data

def load_structure_train_oh():
    X, Y = load_structure_train()
    X = one_hot_encode_material(X)
    return X, Y

def load_structure_test_oh():
    X, Y = load_structure_test()
    X = one_hot_encode_material(X)
    return X, Y

# Validity
def load_validity_train(): return load_dataset_pair("validity", "Predictive_Modeling_Datasets")
def load_validity_test(): return load_dataset_pair_test("validity", "Predictive_Modeling_Datasets")

def load_validity_train_oh():
    X, Y = load_validity_train()
    X = one_hot_encode_material(X)
    return X, Y

def load_validity_test_oh():
    X, Y = load_validity_test()
    X = one_hot_encode_material(X)
    return X, Y

# Usability (continuous)
def load_usability_cont_train(): return load_dataset_pair("usability_cont", "Predictive_Modeling_Datasets", y_ext="tab")
def load_usability_cont_test(): return load_dataset_pair_test("usability_cont", "Predictive_Modeling_Datasets", y_ext="tab")

# CLIP (Y_train is .npy)
def load_clip_train(): return load_dataset_pair("CLIP", "Predictive_Modeling_Datasets", y_train_is_npy=True)
def load_clip_test(): return load_dataset_pair_test("CLIP", "Predictive_Modeling_Datasets")

# ---- Generative modeling dataset functions ----

def load_bike_bench_train():
    path = download_if_missing("Generative_Modeling_Datasets/bike_bench.csv")
    return pd.read_csv(path, index_col=0)

def load_bike_bench_test():
    path = download_if_missing("Generative_Modeling_Datasets/bike_bench_test.csv")
    return pd.read_csv(path, index_col=0)

def load_bike_bench_mixed_modality_train():
    path = download_if_missing("Generative_Modeling_Datasets/bike_bench_mixed_modality.csv")
    return pd.read_csv(path, index_col=0)

def load_bike_bench_mixed_modality_test():
    path = download_if_missing("Generative_Modeling_Datasets/bike_bench_mixed_modality_test.csv")
    return pd.read_csv(path, index_col=0)
