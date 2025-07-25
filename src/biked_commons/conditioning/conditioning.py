import numpy as np
import pandas as pd
import torch


from biked_commons.data_loading import data_loading
from biked_commons.resource_utils import resource_path



# 1a) DEVICE cache for later
_DEVICE_CACHE = {
    "rider": {},    # maps torch.device -> rider‐tensor‐on‐that‐device
    "embed": {},
}

# 1b) Raw CPU tensors (used if you never pass device)
_riders_train_df, _ = data_loading.load_aero_train()
_riders_test_df,  _ = data_loading.load_aero_test()
RIDER_COLS = ['upper_leg','lower_leg','arm_length',
              'torso_length','neck_and_head_length','torso_width']
_RIDER_CPU = {
    "train": torch.tensor(_riders_train_df[RIDER_COLS].values, dtype=torch.float32),
    "test":  torch.tensor(_riders_test_df[RIDER_COLS].values, dtype=torch.float32),
}

_, _embed_train = data_loading.load_clip_train()
_, _embed_test_df  = data_loading.load_clip_test()
_EMBED_CPU = {
    "train": torch.tensor(_embed_train, dtype=torch.float32),
    "test":  torch.tensor(_embed_test_df.values, dtype=torch.float32),
}

# one‐hot on CPU
_USECASE_CPU = torch.eye(3, dtype=torch.float32)

def _get_rider_tensor(split: str, device: torch.device = None):
    if device is None:
        return _RIDER_CPU[split]
    cache = _DEVICE_CACHE["rider"]
    if device not in cache:
        cache[device] = _RIDER_CPU[split].to(device, non_blocking=True)
    return cache[device]

def _get_embed_tensor(split: str, device: torch.device = None):
    if device is None:
        return _EMBED_CPU[split]
    cache = _DEVICE_CACHE["embed"]
    if device not in cache:
        cache[device] = _EMBED_CPU[split].to(device, non_blocking=True)
    return cache[device]

def _get_usecase_tensor(device: torch.device = None):
    if device is None:
        return _USECASE_CPU
    # we can reuse the same 3×3 one‐hot everywhere
    return _USECASE_CPU.to(device, non_blocking=True)

def sample_riders(num_samples: int, split="test",
                  randomize=False, device: torch.device = None):
    data = _get_rider_tensor(split, device)
    N = data.size(0)
    if randomize:
        idx = torch.randint(0, N, (num_samples,), device=device)
    else:
        reps = num_samples // N + 1
        idx = torch.arange(N, device=device).repeat(reps)[:num_samples]
    return data[idx]

def sample_image_embedding(num_samples: int, split="test",
                           randomize=False, device: torch.device = None):
    data = _get_embed_tensor(split, device)
    N = data.size(0)
    if randomize:
        idx = torch.randint(0, N, (num_samples,), device=device)
    else:
        reps = num_samples // N + 1
        idx = torch.arange(N, device=device).repeat(reps)[:num_samples]
    return data[idx]

def sample_use_case(num_samples: int, split=None, randomize=False,
                    device: torch.device = None):
    onehot = _get_usecase_tensor(device)    # shape (3,3)
    if randomize:
        idx = torch.randint(0, 3, (num_samples,), device=device)
    else:
        reps = num_samples // 3 + 1
        idx = torch.arange(3, device=device).repeat(reps)[:num_samples]
    return onehot[idx]


def sample_text(num_samples, split="test", randomize = False):
    # read from .txt data into list of strings, without keeping the newline character
    if split == "test":
        with open(resource_path("text_descriptions/text_descriptions_test.txt"), "r") as f:
            text_data = f.readlines()
    
    elif split == "train":
        with open(resource_path("text_descriptions/text_descriptions_train.txt"), "r") as f:
            text_data = f.readlines()
    else:
        raise ValueError("Invalid split. Choose 'train' or 'test'.")
    #remove newline character from each string in the list
    text_data = [x.strip() for x in text_data]
    #select num_samples from list with replacement
    if randomize:
        sampled_text = np.random.choice(text_data, size=num_samples, replace=True).tolist()
    else:
        #repeat text data until it is long enough
        text_data = text_data * (num_samples // len(text_data) + 1)
        sampled_text = text_data[:num_samples]

    
    return sampled_text
