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
    try:
        with open(file_path, 'r') as f:
            tasks = json.load(f)
        return tasks
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {file_path}")

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
    Save predictions to a submission file in the required format.
    
    Args:
        predictions: Dictionary mapping task IDs to predictions
        file_path: Path where the submission file will be saved
    """
    with open(file_path, 'w') as f:
        json.dump(predictions, f)
