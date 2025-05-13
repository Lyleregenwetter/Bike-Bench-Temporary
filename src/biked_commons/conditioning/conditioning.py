import numpy as np
import pandas as pd
import torch


from biked_commons.resource_utils import split_datasets_path

def sample_riders(num_samples, split = "test", randomize = False):
    # Sample random riders from the rider data
    if split == "test":
        rider_data = pd.read_csv(split_datasets_path("aero_X_test.csv"), index_col=0)
        rider_data = rider_data[['upper_leg', 'lower_leg', 'arm_length', 'torso_length', 'neck_and_head_length', 'torso_width']]
    elif split == "train":
        rider_data = pd.read_csv(split_datasets_path("aero_X_train.csv"), index_col=0)
        rider_data = rider_data[['upper_leg', 'lower_leg', 'arm_length', 'torso_length', 'neck_and_head_length', 'torso_width']]
    else:
        raise ValueError("Invalid split. Choose 'train' or 'test'.")
    if randomize:
        sampled_riders = rider_data.sample(n=num_samples, replace=True).values
    else:
        rider_data = pd.concat([rider_data] * (num_samples // len(rider_data) + 1), ignore_index=True)
        sampled_riders = rider_data.iloc[:num_samples].values
    return torch.tensor(sampled_riders, dtype=torch.float32)

def sample_use_case(num_samples, split=None, randomize = False):    
    # Randomly pick indices 0, 1 or 2

    if randomize:
        # Randomly pick indices 0, 1 or 2
        idx = np.random.choice(3, size=num_samples, replace=True)
    else:
        # Repeat the indices 0, 1, 2 until it is long enough
        idx = np.tile(np.arange(3), num_samples // 3 + 1)[:num_samples]
    
    # Convert to one-hot
    onehots = np.eye(3, dtype=int)[idx]
    
    return torch.tensor(onehots, dtype=torch.float32)

def sample_text(num_samples, split="test", randomize = False):
    # read from .txt data into list of strings, without keeping the newline character
    if split == "test":
        with open(split_datasets_path("text_descriptions_test.txt"), "r") as f:
            text_data = f.readlines()
    
    elif split == "train":
        with open(split_datasets_path("text_descriptions_train.txt"), "r") as f:
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

def sample_image_embedding(num_samples, split="test", randomize = False):
    # Sample random riders from the rider data
    if split == "test":
        embeddings = np.load(split_datasets_path("CLIP_Y_test.npy"))
    elif split == "train":
        embeddings = np.load(split_datasets_path("CLIP_Y_train.npy"))
    else:
        raise ValueError("Invalid split. Choose 'train' or 'test'.")
    
    # Sample random images from the image data
    if randomize:
        sampled_indices = np.random.choice(len(embeddings), size=num_samples, replace=True)
        sampled_images = embeddings[sampled_indices]
    else:
        embeddings = np.tile(embeddings, (num_samples // len(embeddings) + 1, 1))
        sampled_images = embeddings[:num_samples]
    
    return torch.tensor(sampled_images, dtype=torch.float32)