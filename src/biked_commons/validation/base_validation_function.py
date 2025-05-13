from abc import abstractmethod, ABC
from typing import List

import pandas as pd
import torch


# TODO: write validation functions to [optionally] be able to grab values from the default bike when not found?

class ValidationFunction(ABC):
    @abstractmethod
    def friendly_name(self) -> str:
        """
        Should return a user-friendly and easily comprehensible name for the validation in question.
        """
        pass

    @abstractmethod
    def variable_names(self) -> List[str]:
        """
        Should return a list of variable names used in the validation.
        """
        pass

    @abstractmethod
    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        """
        Should return a PyTorch tensor with shape (len(designs), 1) or (len(designs),).
        The values in the tensor represent validity. 1 is invalid, 0 is valid.
        """
        pass


def construct_tensor_validator(validation_functions: List[ValidationFunction], column_names: List[str]):
    """
    Constructs a function that applies multiple validation functions to a PyTorch tensor of designs.
    
    Parameters:
        validation_functions (List[ValidationFunction]): List of validation function instances.
        column_names (List[str]): List of column names in the same order as tensor features.

    Returns:
        A function that takes a PyTorch tensor of designs and returns a PyTorch tensor of validation results.
    """

    column_names = list(column_names)

    all_return_names = [v.friendly_name() for v in validation_functions]

    def validate_tensor(designs: torch.Tensor) -> torch.Tensor:
        """
        Applies the validation functions to the given tensor and returns a tensor of results.
        
        Parameters:
            designs (torch.Tensor): A tensor where each row represents a design.

        Returns:
            torch.Tensor: A tensor of shape (n, v), where:
                - Rows correspond to designs.
                - Columns correspond to validation function results.
                - Values: 1 indicates invalid, 0 indicates valid.
        """
        n = designs.shape[0]
        v = len(validation_functions)

        # Initialize results tensor with zeros (default: valid)
        results_tensor = torch.zeros((n, v), dtype=torch.float32, device=designs.device)

        for i, validation_function in enumerate(validation_functions):
            # try:


            # Get the indices of the required variables for this function
            var_indices = [column_names.index(var) for var in validation_function.variable_names()]

            # Extract the relevant slices from the tensor
            sliced_designs = designs[:, var_indices]

            # Apply validation
            res = validation_function.validate(sliced_designs)  # Expected to return a torch.Tensor

            # Store results
            results_tensor[:, i] = res.flatten()


            # except Exception as e:
            #     print(f"Validation function [{validation_function.friendly_name()}] encountered exception [{e}]")
            #     results_tensor[:, i] = 1  # Mark all designs as invalid in case of failure

        return results_tensor

    return validate_tensor, all_return_names


def construct_dataframe_validator(validation_functions: List[ValidationFunction]):
    """
    Constructs a function that applies multiple validation functions to a Pandas DataFrame of designs.
    
    Parameters:
        validation_functions (List[ValidationFunction]): List of validation function instances.

    Returns:
        A function that takes a Pandas DataFrame of designs and returns a DataFrame of validation results.
    """

    # First, construct the tensor-based validator (this one doesn't need column mapping)
    def validate_dataframe(designs: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the DataFrame to a tensor, applies validation, and converts the result back to a DataFrame.
        
        Parameters:
            designs (pd.DataFrame): A DataFrame where each row represents a design.

        Returns:
            pd.DataFrame: A DataFrame of shape (n, v), where:
                - Rows correspond to designs (original DataFrame index is preserved).
                - Columns correspond to validation function names.
                - Values: 1 indicates invalid, 0 indicates valid.
        """
        # Convert DataFrame to a PyTorch tensor (float32)
        designs_tensor = torch.tensor(designs.to_numpy(), dtype=torch.float32)

        # Use the tensor validator (construct it dynamically based on DataFrame columns)
        tensor_validator, all_return_names = construct_tensor_validator(validation_functions, list(designs.columns))
        results_tensor = tensor_validator(designs_tensor)

        # Convert results back to a DataFrame
        results_df = pd.DataFrame(
            results_tensor.numpy(),  # Convert tensor to NumPy
            columns=all_return_names,
            index=designs.index  # Preserve original index
        )

        return results_df

    return validate_dataframe
