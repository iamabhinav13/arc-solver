import json
import os
import numpy as np
from typing import Dict, List, Any, Optional

def load_tasks(file_path: str) -> Dict[str, Any]:
    """
    Load and parse ARC tasks from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing tasks
        
    Returns:
        Dictionary mapping task IDs to task data
    """
    # Handle the case where only the filename is provided
    if not os.path.exists(file_path) and not file_path.startswith('/'):
        potential_paths = [
            os.path.join('data', file_path),                 # Check in data directory
            file_path,                                       # Check in current directory
            os.path.join('..', file_path)                    # Check in parent directory
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                file_path = path
                break
    
    try:
        with open(file_path, 'r') as f:
            tasks = json.load(f)
        return tasks
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}. Make sure the dataset files are in the 'data' directory.")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {file_path}")

def load_solutions(file_path: str) -> Dict[str, List[np.ndarray]]:
    """
    Load solutions from an ARC Prize solution file.
    
    Args:
        file_path: Path to the JSON file containing solutions
        
    Returns:
        Dictionary mapping task IDs to solution grids
    """
    # Handle the case where only the filename is provided
    if not os.path.exists(file_path) and not file_path.startswith('/'):
        potential_paths = [
            os.path.join('data', file_path),                 # Check in data directory
            file_path,                                       # Check in current directory
            os.path.join('..', file_path)                    # Check in parent directory
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                file_path = path
                break
    
    try:
        with open(file_path, 'r') as f:
            solutions_raw = json.load(f)
            
        # Convert solutions to numpy arrays
        solutions = {}
        for task_id, task_solutions in solutions_raw.items():
            solutions[task_id] = [np.array(grid) for grid in task_solutions]
            
        return solutions
    except FileNotFoundError:
        raise FileNotFoundError(f"Solutions file not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in solutions file: {file_path}")

def get_task_grids(task: Dict[str, Any]) -> Dict[str, List[np.ndarray]]:
    """
    Extract input and output grids from a task.
    
    Args:
        task: Task data
        
    Returns:
        Dictionary with 'train_inputs', 'train_outputs', and 'test_inputs' keys
    """
    train_inputs = []
    train_outputs = []
    
    # Process training pairs
    train_pairs = task.get('train', [])
    
    for pair in train_pairs:
        train_inputs.append(np.array(pair['input']))
        train_outputs.append(np.array(pair['output']))
    
    # Process test inputs
    test_inputs = []
    test_data = task.get('test', [])
    
    # Handle both single test and multiple tests
    if isinstance(test_data, dict):
        test_inputs.append(np.array(test_data['input']))
    else:
        for test in test_data:
            test_inputs.append(np.array(test['input']))
    
    return {
        'train_inputs': train_inputs,
        'train_outputs': train_outputs,
        'test_inputs': test_inputs
    }

def save_submission(predictions: Dict[str, List[Dict[str, List[List[int]]]]], 
                    file_path: str = 'submission.json') -> None:
    """
    Save predictions to a submission file in the required format for the ARC Prize competition.
    
    Args:
        predictions: Dictionary mapping task IDs to predictions
        file_path: Path where the submission file will be saved
    """
    # Ensure the predictions match the expected format:
    # {
    #   "task_id": [
    #     {
    #       "attempt_1": [[0, 0], ...],
    #       "attempt_2": [[0, 0], ...]
    #     },
    #     ...
    #   ],
    #   ...
    # }
    
    with open(file_path, 'w') as f:
        json.dump(predictions, f, indent=2)
