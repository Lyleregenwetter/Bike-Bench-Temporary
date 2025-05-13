import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score, r2_score, mean_squared_error
from biked_commons.prediction.loaders import one_hot_encode_material
from sklearn.metrics.pairwise import cosine_similarity

from biked_commons.resource_utils import resource_path, split_datasets_path

def evaluate_validity(model, preprocessing_fn, device="cpu"):
    X_test = pd.read_csv(split_datasets_path('validity_X_test.csv'), index_col=0)    
    Y_test = pd.read_csv(split_datasets_path('validity_Y_test.csv'), index_col=0)
    X_test = one_hot_encode_material(X_test)
    X_test_tensor = torch.tensor(X_test.values.astype(float), dtype=torch.float32).to(device)
    X_test_tensor = preprocessing_fn(X_test_tensor)
    predictions = model(X_test_tensor).detach().cpu().numpy()
    predictions = predictions>= 0.5
    return f1_score(Y_test, predictions)

def evaluate_structure(model, preprocessing_fn, device="cpu"):
    Y_test = pd.read_csv(split_datasets_path('structure_Y_test.csv'), index_col=0)
    X_test = pd.read_csv(split_datasets_path('structure_X_test.csv'), index_col=0)
    X_test = one_hot_encode_material(X_test)
    X_test_tensor = torch.tensor(X_test.values.astype(float), dtype=torch.float32).to(device)
    X_test_tensor = preprocessing_fn(X_test_tensor)
    predictions = model(X_test_tensor).detach().cpu().numpy()
    return r2_score(Y_test, predictions)

def evaluate_aero(model, preprocessing_fn, device="cpu"):
    Y_test = pd.read_csv(split_datasets_path('aero_Y_test.csv'), index_col=0)
    X_test = pd.read_csv(split_datasets_path('aero_X_test.csv'), index_col=0)
    X_test_tensor = torch.tensor(X_test.values.astype(float), dtype=torch.float32).to(device)
    X_test_tensor = preprocessing_fn(X_test_tensor)
    predictions = model(X_test_tensor).detach().cpu().numpy()
    return r2_score(Y_test, predictions)

def evaluate_usability(model, preprocessing_fn, device="cpu", target_type='cont'):
    if target_type == 'cont':
        X_train = pd.read_csv(split_datasets_path('usability_cont_X_test.csv'), index_col=0)
        Y_train = pd.read_csv(split_datasets_path('usability_cont_Y_test.csv'), index_col=0)
        X_train_tensor = torch.tensor(X_train.values.astype(float), dtype=torch.float32).to(device)
        X_train_tensor = preprocessing_fn(X_train_tensor)
        predictions = model(X_train_tensor).detach().cpu().numpy()
        return r2_score(Y_train, predictions)
    elif target_type == 'binary':
        X_train = pd.read_csv(split_datasets_path('usability_binary_X_test.csv'), index_col=0)
        Y_train = pd.read_csv(split_datasets_path('usability_binary_Y_test.csv'), index_col=0)
        X_train_tensor = torch.tensor(X_train.values.astype(float), dtype=torch.float32).to(device)
        X_train_tensor = preprocessing_fn(X_train_tensor)
        predictions = model(X_train_tensor).detach().cpu().numpy()
        predictions = predictions >= 0.5
        return f1_score(Y_train, predictions)

def evaluate_clip(model, preprocessing_fn, device="cpu"):
    Y_test = np.load(split_datasets_path('split_datasets/CLIP_Y_test.npy'))
    X_test = pd.read_csv(split_datasets_path('CLIP_X_test.csv'), index_col=0)
    X_test_tensor = torch.tensor(X_test.values.astype(float), dtype=torch.float32).to(device)
    X_test_tensor = preprocessing_fn(X_test_tensor)
    predictions = model(X_test_tensor).detach().cpu().numpy()

    cosine_sim = cosine_similarity(predictions, Y_test)
    diag = np.diag(cosine_sim)
    worse_than_diag = cosine_sim <= diag[:, np.newaxis]
    matchperc = np.mean(worse_than_diag)
    print(f"Predicted embedding more similar to GT than : {100 * matchperc:.2f}% of test set designs, on average.")

    return mean_squared_error(Y_test, predictions)