from abc import abstractmethod, ABC
from typing import List
import torch
import pandas as pd
import numpy as np
import pygmo as pg
from sklearn.preprocessing import StandardScaler
import os
from biked_commons.conditioning import conditioning
from biked_commons.resource_utils import resource_path
from biked_commons.data_loading import data_loading
from biked_commons.design_evaluation.design_evaluation import construct_tensor_evaluator, EvaluationFunction
from biked_commons.transformation import one_hot_encoding


class ScoringFunction(ABC):
    def __init__(self, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def return_names(self) -> List[str]:
        pass

    @abstractmethod
    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        pass

def compute_ref_point(ref_scores, reduction):
    ref_scores[np.isnan(ref_scores)] = -float("inf")
    if reduction == "max":
        ref_point = np.max(ref_scores, axis=0)
    elif reduction == "meanabs":
        ref_point = np.mean(np.abs(ref_scores), axis=0)
    return ref_point

def recompute_ref_point(evaluator, eval_names, path, reduction, device):
    print("Calculating reference point for scoring functions...")
    data = pd.read_csv(data_loading.load_bike_bench_test(), index_col=0)
    num_data = data.shape[0]
    rider_condition = conditioning.sample_riders(num_data, split="test")
    use_case_condition = conditioning.sample_use_case(num_data, split="test")
    embedding = conditioning.sample_image_embedding(num_data, split="test")

    condition = {"Rider": rider_condition, "Use Case": use_case_condition, "Embedding": embedding}
    scores = evaluator(torch.tensor(data.values, dtype=torch.float32, device=device), condition)
    objective_scores = scores.detach().cpu().numpy()
    ref_point = compute_ref_point(objective_scores, reduction)
    df = pd.Series(ref_point, index=eval_names)
    df.to_csv(path, header=False)
    return df

def get_ref_point(evaluator, objective_names, eval_names, reduction = "max", device = "cpu"):
    if reduction=="max":
        path = resource_path("misc/ref_point.csv")
    elif reduction=="meanabs":
        path = resource_path("misc/default_weights.csv")
    else:
        raise ValueError("Invalid reduction method. Use 'max' or 'meanabs'.")
    if not os.path.exists(path):
        ref_point_df = recompute_ref_point(evaluator, eval_names, path, reduction, device)
    else:
        ref_point_df = pd.read_csv(path, index_col=0, header=None)
        ref_point_columns = ref_point_df.index.values
        if not np.all(np.isin(objective_names, ref_point_columns)):
            print("Reference point does not include all objective names. Recomputing...")
            ref_point_df = recompute_ref_point(evaluator, eval_names, path, reduction, device)
    ref_point_df = ref_point_df.loc[objective_names]
    ref_point = ref_point_df.values.flatten()
    return ref_point

class Hypervolume(ScoringFunction):
    def __init__(self):
        super().__init__()

    def return_names(self) -> List[str]:
        return ["Hypervolume"]

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point):
        
        validity_mask = np.all(constraint_scores <= 0, axis=1)
        valid_objective_scores = objective_scores[validity_mask]
        if valid_objective_scores.size == 0:
            return 0.0
        valid_objective_scores[np.isnan(valid_objective_scores)] = float("inf")
        valid_objective_scores = valid_objective_scores/obj_ref_point
        valid_objective_scores = np.clip(valid_objective_scores, a_min=0, a_max=1)
        scaled_ref_point = np.ones_like(obj_ref_point)

        hv = pg.hypervolume(valid_objective_scores)
        hv_value = hv.compute(ref_point=scaled_ref_point)
        return hv_value

class ConstraintSatisfactionRate(ScoringFunction):
    def __init__(self):
        super().__init__()

    def return_names(self) -> List[str]:
        return ["Constraint Satisfaction Rate"]
    
    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point):
        return np.mean(np.all(constraint_scores <=0, axis=1))


class MMD(ScoringFunction): 

    def __init__(self, batch_size = 1024, gamma=None):
        super().__init__()
        raw_ref  = data_loading.load_bike_bench_test().values.astype(np.float32)
        
        self.scaler = StandardScaler()
        self.scaler.fit(raw_ref)
        self.reference_designs = self.scaler.transform(raw_ref)

        self.batch_size = batch_size

        if gamma is None:
            gamma = self.compute_gamma(self.reference_designs)
        self.gamma = gamma
        

    def return_names(self) -> List[str]:
        return ["Maximum Mean Discrepancy"]

    def compute_gamma(self, ref: np.ndarray) -> float:
        dists = np.sum((ref[:, None, :] - ref[None, :, :])**2, axis=2)
        med = np.median(dists)
        return 1.0 / (2 * med) if med > 0 else 1.0

    def rbf_kernel_sum(self, A: np.ndarray, B: np.ndarray, gamma: float) -> float:
        """
        Compute sum_{i,j} exp(-gamma * ||A[i] - B[j]||^2)
        by blocking through rows of A and B in chunks of size batch_size.
        """
        total = 0.0
        for i in range(0, A.shape[0], self.batch_size):
            Ai = A[i : i + self.batch_size]
            for j in range(0, B.shape[0], self.batch_size):
                Bj = B[j : j + self.batch_size]
                # compute squared‐distances of shape (len(Ai), len(Bj))
                D2 = np.sum((Ai[:, None, :] - Bj[None, :, :])**2, axis=2)
                total += np.exp(-gamma * D2).sum()
        return total

    def mmd(self, gen: np.ndarray, ref: np.ndarray) -> float:
        K_GG = self.rbf_kernel_sum(gen, gen, self.gamma)
        K_RR = self.rbf_kernel_sum(ref, ref, self.gamma)
        K_GR = self.rbf_kernel_sum(gen, ref, self.gamma)

        n, m = gen.shape[0], ref.shape[0]
        return (K_GG / (n * n)) + (K_RR / (m * m)) - (2 * K_GR / (n * m))

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point):
        scaled_designs = self.scaler.transform(designs)
        return self.mmd(scaled_designs, self.reference_designs)
    

class MinimumObjective(ScoringFunction):
    def __init__(self):
        super().__init__()

    def return_names(self) -> List[str]:
        return self.names

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point):
        self.names = [f"Min Objective Score: {name}" for name in objective_names]
        validity_mask = np.all(constraint_scores <= 0, axis=1)
        valid_objective_scores = objective_scores[validity_mask]
        if valid_objective_scores.size == 0:
            return np.ones_like(objective_scores[0]) * obj_ref_point
        minscores = np.min(valid_objective_scores, axis=0)
        return minscores
    
class MeanObjective(ScoringFunction):
    def __init__(self):
        super().__init__()

    def return_names(self) -> List[str]:
        return self.names

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point):
        self.names = [f"Mean Objective Score: {name}" for name in objective_names]
        validity_mask = np.all(constraint_scores <= 0, axis=1)
        valid_objective_scores = objective_scores[validity_mask]
        if valid_objective_scores.size == 0:
            return np.ones_like(objective_scores[0]) * obj_ref_point
        meanscores = np.mean(valid_objective_scores, axis=0)
        return meanscores
    
class ConstraintViolationRate(ScoringFunction):
    def __init__(self):
        super().__init__()

    def return_names(self) -> List[str]:
        return self.names

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point):
        self.names = [f"Constraint Violation Rate: {name}" for name in constraint_names]
        validity_boolean = constraint_scores > 0
        return np.mean(validity_boolean, axis=0)
    
class MeanConstraintViolationMagnitude(ScoringFunction):
    def __init__(self):
        super().__init__()

    def return_names(self) -> List[str]:
        return self.names

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point):
        self.names = [f"Mean Constraint Violation Magnitude: {name}" for name in constraint_names]
        constraint_scores = np.clip(constraint_scores, a_min=0, a_max=None)
        meanscores = np.mean(constraint_scores, axis=0)
        return meanscores

def construct_scorer(scoring_functions: List[ScoringFunction], evaluation_functions: List[EvaluationFunction], column_names: List[str], device: str = "cpu") -> callable:

    evaluator, requirement_names, requirement_types = construct_tensor_evaluator(evaluation_functions, column_names, device=device)
    requirement_names = np.array(requirement_names)
    isobjective = torch.tensor(requirement_types) == 1
    objective_names = requirement_names[isobjective]
    constraint_names = requirement_names[~isobjective]

    obj_ref_point = get_ref_point(evaluator, objective_names, requirement_names, "max", device) #1D numpy array
    def scorer(designs: torch.Tensor, condition: dict = {}) -> pd.Series:
        device = designs.device
        designs = designs.detach().cpu().numpy()
        score_names = []
        scores = []
        designs_df = pd.DataFrame(designs, columns=column_names)
        designs_reverse_oh = one_hot_encoding.decode_to_mixed(designs_df)
        designs_continuous_mapped = one_hot_encoding.encode_to_continuous(designs_reverse_oh)
        designs_mapped_tens = torch.tensor(designs_continuous_mapped.values, dtype=torch.float32).to(device)
        evaluation_scores = evaluator(designs_mapped_tens, condition)
        objective_scores = evaluation_scores[:, isobjective].detach().cpu().numpy()
        ref_point_exp = np.expand_dims(obj_ref_point, axis=0)
        ref_point_exp = np.repeat(ref_point_exp, objective_scores.shape[0], axis=0)
        objective_scores[np.isnan(objective_scores)] = ref_point_exp[np.isnan(objective_scores)]
        constraint_scores = evaluation_scores[:, ~isobjective].detach().cpu().numpy()
        
        for scoring_function in scoring_functions:    
            raw = scoring_function.evaluate(designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point)

            arr = np.atleast_1d(raw)

            names = scoring_function.return_names()

            for n, val in zip(names, arr):
                score_names.append(n)
                scores.append(val)
        scores = np.array(scores)
        scores = pd.Series(scores, index=score_names)
        return scores
    return scorer

MainScores: List[ScoringFunction] = [
    Hypervolume(),
    ConstraintSatisfactionRate(),
    MMD(),
]

DetailedScores: List[ScoringFunction] = [
    MinimumObjective(),
    MeanObjective(),
    ConstraintViolationRate(),
    MeanConstraintViolationMagnitude(),
]



