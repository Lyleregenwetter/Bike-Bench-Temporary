import os
import torch
import pandas as pd
from biked_commons.design_evaluation.scoring import construct_scorer, MainScores, DetailedScores
from biked_commons.design_evaluation.design_evaluation import get_standard_evaluations
from biked_commons.conditioning import conditioning

def get_condition_by_idx(idx=0):
    rider_condition = conditioning.sample_riders(10, split="test")
    use_case_condition = conditioning.sample_use_case(10, split="test")
    image_embeddings = conditioning.sample_image_embedding(10, split="test")
    condition = {"Rider": rider_condition[idx], "Use Case": use_case_condition[idx], "Embedding": image_embeddings[idx]}
    return condition

def get_conditions_10k():
    rider_condition = conditioning.sample_riders(10000, split="test")
    use_case_condition = conditioning.sample_use_case(10000, split="test")
    image_embeddings = conditioning.sample_image_embedding(10000, split="test")
    conditions = {"Rider": rider_condition, "Use Case": use_case_condition, "Embedding": image_embeddings}
    return conditions

def evaluate_uncond(result_tens, name, cond_idx, data_columns, device, save=True):

    condition = get_condition_by_idx(cond_idx)

    result_dir = os.path.join("results", "unconditional", f"cond_{cond_idx}", name)
    os.makedirs(result_dir, exist_ok=True)
    
    main_scorer = construct_scorer(MainScores, get_standard_evaluations(device), data_columns)
    detailed_scorer = construct_scorer(DetailedScores, get_standard_evaluations(device), data_columns)

    main_scores = main_scorer(result_tens, condition)
    
    detailed_scores = detailed_scorer(result_tens, condition)
    
    if save:
        result_tens = result_tens.cpu()
        torch.save(result_tens, os.path.join(result_dir, "result_tens.pt"))
        main_scores.to_csv(os.path.join(result_dir, "main_scores.csv"), index_label=False, header=False)
        detailed_scores.to_csv(os.path.join(result_dir, "detailed_scores.csv"), index_label=False, header=False)
    return main_scores, detailed_scores

def evaluate_cond(result_tens, name, data_columns, device, indices = range(10000), save=True):
    condition = get_conditions_10k()

    condition = {"Rider": condition["Rider"][indices], "Use Case": condition["Use Case"][indices], "Embedding": condition["Embedding"][indices]}

    result_dir = os.path.join("results", "conditional", name)
    os.makedirs(result_dir, exist_ok=True)

    main_scorer = construct_scorer(MainScores, get_standard_evaluations(device), data_columns, device)
    detailed_scorer = construct_scorer(DetailedScores, get_standard_evaluations(device), data_columns, device)

    main_scores = main_scorer(result_tens, condition)
    detailed_scores = detailed_scorer(result_tens, condition)

    if save:
        result_tens = result_tens.cpu()
        torch.save(result_tens, os.path.join(result_dir, "result_tens.pt"))
        main_scores.to_csv(os.path.join(result_dir, "main_scores.csv"), index_label=False, header=False)
        detailed_scores.to_csv(os.path.join(result_dir, "detailed_scores.csv"), index_label=False, header=False)

    return main_scores, detailed_scores


def create_score_report_conditional():
    """
    Looks through the results folder and creates a score report for each conditional result.
    """
    all_scores = []
    result_dir = os.path.join("results", "conditional")
    for name in os.listdir(result_dir):
        if os.path.isdir(os.path.join(result_dir, name)):
            main_scores = pd.read_csv(os.path.join(result_dir, name, "main_scores.csv"), header=None)
            main_scores.columns = ["Metric", "Score"]
            main_scores["Model"] = name
            all_scores.append(main_scores)
    all_scores = pd.concat(all_scores, axis=0)
    #make metric names the three columns, make models the rows
    all_scores = all_scores.pivot(index="Model", columns="Metric", values="Score")
    #drop the index name and the column name
    all_scores.columns.name = None
    all_scores.index.name = None
    
    return all_scores

def create_score_report_unconditional():
    """
    Looks through the results folder and creates a score report for each unconditional result.
    """
    all_scores = []
    result_dir = os.path.join("results", "unconditional")
    for i in range(10):
        c_dir = os.path.join(result_dir, f"cond_{i}")
        for name in os.listdir(c_dir):
            dirname = os.path.join(c_dir, name)
            if os.path.isdir(dirname):
                main_scores = pd.read_csv(os.path.join(dirname, "main_scores.csv"), header=None)
                main_scores.columns = ["Metric", "Score"]
                main_scores["Model"] = name
                main_scores["Condition"] = i
                all_scores.append(main_scores)
    all_scores = pd.concat(all_scores, axis=0)
    #average over condition 
    all_scores = all_scores.groupby(["Model", "Metric"]).mean().reset_index()
    #make metric names the three columns, make models the rows
    all_scores = all_scores.pivot(index="Model", columns="Metric", values="Score")
    #drop the index name and the column name
    all_scores.columns.name = None
    all_scores.index.name = None
    return all_scores
