import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Set
import random

from reasoning.pattern_detection import (
    extract_colors, count_color_occurrences, detect_color_mapping,
    find_objects, detect_symmetry, detect_pattern_repetition, detect_size_changes
)
from reasoning.transformations import (
    apply_color_mapping, rotate_grid, flip_grid, scale_grid,
    crop_grid, pad_grid, replace_color, apply_logical_operation,
    apply_mathematical_operation, shift_grid, transpose_grid,
    fill_pattern, get_transformation_function
)

def pattern_matching_reasoning(task: Dict[str, Any]) -> List[np.ndarray]:
    """
    Apply pattern matching reasoning to solve a task.
    
    Args:
        task: Task data
        
    Returns:
        List of predicted output grids
    """
    # Extract input/output pairs from the task
    train_inputs = []
    train_outputs = []
    
    for pair in task.get('train', []):
        train_inputs.append(np.array(pair['input']))
        train_outputs.append(np.array(pair['output']))
    
    # Extract test inputs
    test_inputs = []
    test_data = task.get('test', [])
    
    if isinstance(test_data, dict):
        test_inputs.append(np.array(test_data['input']))
    else:
        for test in test_data:
            test_inputs.append(np.array(test['input']))
    
    # Generate predictions
    predictions = []
    
    for test_input in test_inputs:
        # Try to find a consistent color mapping
        color_mappings = []
        
        for i in range(len(train_inputs)):
            mapping = detect_color_mapping(train_inputs[i], train_outputs[i])
            if mapping:
                color_mappings.append(mapping)
        
        # If we found consistent color mappings, apply them
        if color_mappings:
            prediction = apply_color_mapping(test_input, color_mappings[0])
            predictions.append(prediction)
            continue
        
        # Try to find size changes
        size_changes = []
        
        for i in range(len(train_inputs)):
            changes = detect_size_changes(train_inputs[i], train_outputs[i])
            size_changes.append(changes)
        
        # Check if height and width scaling is consistent
        consistent_height_scale = all(abs(changes['height_scale'] - size_changes[0]['height_scale']) < 0.01 for changes in size_changes)
        consistent_width_scale = all(abs(changes['width_scale'] - size_changes[0]['width_scale']) < 0.01 for changes in size_changes)
        
        if consistent_height_scale and consistent_width_scale:
            # Apply the scaling to test input
            h_scale = size_changes[0]['height_scale']
            w_scale = size_changes[0]['width_scale']
            
            # If scaling is integer, use scale_grid function
            if h_scale.is_integer() and w_scale.is_integer() and h_scale == w_scale and h_scale > 0:
                prediction = scale_grid(test_input, int(h_scale))
                predictions.append(prediction)
                continue
        
        # Try simple transformations
        transformations = [
            ("rotate_90", lambda g: rotate_grid(g, 1)),
            ("rotate_180", lambda g: rotate_grid(g, 2)),
            ("rotate_270", lambda g: rotate_grid(g, 3)),
            ("flip_horizontal", lambda g: flip_grid(g, 1)),
            ("flip_vertical", lambda g: flip_grid(g, 0)),
            ("transpose", transpose_grid)
        ]
        
        for name, transform_fn in transformations:
            matches = 0
            
            for i in range(len(train_inputs)):
                transformed = transform_fn(train_inputs[i])
                if transformed.shape == train_outputs[i].shape and np.array_equal(transformed, train_outputs[i]):
                    matches += 1
            
            if matches == len(train_inputs):
                prediction = transform_fn(test_input)
                predictions.append(prediction)
                break
        
        if len(predictions) == len(test_inputs):
            continue
        
        # If we haven't found a solution, try a simple backup solution
        # Just repeat the last training output
        if train_outputs:
            predictions.append(train_outputs[-1])
    
    # Ensure we have at least 2 predictions for each test input
    while len(predictions) < len(test_inputs) * 2:
        predictions.append(predictions[-1].copy() if predictions else np.array([[0]]))
    
    return predictions[:2]  # Return first 2 predictions

def color_transformation_reasoning(task: Dict[str, Any]) -> List[np.ndarray]:
    """
    Apply color transformation reasoning to solve a task.
    
    Args:
        task: Task data
        
    Returns:
        List of predicted output grids
    """
    # Extract input/output pairs from the task
    train_inputs = []
    train_outputs = []
    
    for pair in task.get('train', []):
        train_inputs.append(np.array(pair['input']))
        train_outputs.append(np.array(pair['output']))
    
    # Extract test inputs
    test_inputs = []
    test_data = task.get('test', [])
    
    if isinstance(test_data, dict):
        test_inputs.append(np.array(test_data['input']))
    else:
        for test in test_data:
            test_inputs.append(np.array(test['input']))
    
    # Generate predictions
    predictions = []
    
    for test_input in test_inputs:
        color_predictions = []
        
        # Try color mapping
        for i in range(len(train_inputs)):
            mapping = detect_color_mapping(train_inputs[i], train_outputs[i])
            
            if mapping and len(mapping) > 0:
                prediction = apply_color_mapping(test_input, mapping)
                color_predictions.append(prediction)
        
        # Try color inversion (9 - color)
        inversion_matches = 0
        for i in range(len(train_inputs)):
            inverted = 9 - train_inputs[i]
            if inverted.shape == train_outputs[i].shape and np.array_equal(inverted, train_outputs[i]):
                inversion_matches += 1
        
        if inversion_matches == len(train_inputs):
            prediction = 9 - test_input
            color_predictions.append(prediction)
        
        # Try color cycling (color + 1) % 10
        cycling_matches = 0
        for i in range(len(train_inputs)):
            cycled = (train_inputs[i] + 1) % 10
            if cycled.shape == train_outputs[i].shape and np.array_equal(cycled, train_outputs[i]):
                cycling_matches += 1
        
        if cycling_matches == len(train_inputs):
            prediction = (test_input + 1) % 10
            color_predictions.append(prediction)
        
        # If we found any color transformations, add them to predictions
        if color_predictions:
            predictions.extend(color_predictions[:2])
        else:
            # If no color transformation was found, try to keep the shape but replace colors
            # with most common ones from training outputs
            output_colors = set()
            for output in train_outputs:
                output_colors.update(extract_colors(output))
            
            if output_colors:
                # Apply most common colors from training outputs
                new_grid = np.zeros_like(test_input)
                input_colors = list(extract_colors(test_input))
                output_colors_list = list(output_colors)
                
                # Create a simple mapping
                color_map = {}
                for i, color in enumerate(input_colors):
                    if i < len(output_colors_list):
                        color_map[color] = output_colors_list[i]
                    else:
                        color_map[color] = output_colors_list[0]
                
                prediction = apply_color_mapping(test_input, color_map)
                predictions.append(prediction)
            else:
                # Default prediction
                predictions.append(test_input.copy())
    
    # Ensure we have at least 2 predictions for each test input
    while len(predictions) < len(test_inputs) * 2:
        if len(predictions) > 0:
            predictions.append(predictions[-1].copy())
        else:
            predictions.append(test_inputs[0].copy())
    
    return predictions[:2]  # Return first 2 predictions

def shape_detection_reasoning(task: Dict[str, Any]) -> List[np.ndarray]:
    """
    Apply shape detection reasoning to solve a task.
    
    Args:
        task: Task data
        
    Returns:
        List of predicted output grids
    """
    # Extract input/output pairs from the task
    train_inputs = []
    train_outputs = []
    
    for pair in task.get('train', []):
        train_inputs.append(np.array(pair['input']))
        train_outputs.append(np.array(pair['output']))
    
    # Extract test inputs
    test_inputs = []
    test_data = task.get('test', [])
    
    if isinstance(test_data, dict):
        test_inputs.append(np.array(test_data['input']))
    else:
        for test in test_data:
            test_inputs.append(np.array(test['input']))
    
    # Generate predictions
    predictions = []
    
    for test_input in test_inputs:
        shape_predictions = []
        
        # Try to find objects in test input
        test_objects = find_objects(test_input)
        
        # Try symmetry operations
        input_symmetries = []
        output_symmetries = []
        
        for i in range(len(train_inputs)):
            input_sym = detect_symmetry(train_inputs[i])
            output_sym = detect_symmetry(train_outputs[i])
            input_symmetries.append(input_sym)
            output_symmetries.append(output_sym)
        
        # Check if output symmetry is consistent but different from input
        symmetry_transform = False
        for sym_type in ['horizontal', 'vertical', 'diagonal1', 'diagonal2']:
            all_output_sym = all(sym[sym_type] for sym in output_symmetries if sym_type in sym)
            all_input_diff = all(not sym[sym_type] for sym in input_symmetries if sym_type in sym)
            
            if all_output_sym and all_input_diff:
                symmetry_transform = True
                
                # Apply symmetry transformation
                if sym_type == 'horizontal':
                    prediction = flip_grid(test_input, 1)
                elif sym_type == 'vertical':
                    prediction = flip_grid(test_input, 0)
                elif sym_type == 'diagonal1':
                    prediction = transpose_grid(test_input)
                elif sym_type == 'diagonal2':
                    prediction = rotate_grid(transpose_grid(test_input), 2)
                
                shape_predictions.append(prediction)
        
        # Try pattern repetition
        pattern_matches = 0
        pattern_results = []
        
        for i in range(len(train_inputs)):
            in_pattern = detect_pattern_repetition(train_inputs[i])
            out_pattern = detect_pattern_repetition(train_outputs[i])
            
            if in_pattern and out_pattern:
                pattern_matches += 1
                pattern_results.append((in_pattern, out_pattern))
        
        if pattern_matches > 0 and pattern_matches == len(train_inputs):
            # Try to apply similar pattern transformation
            in_pattern, out_pattern = pattern_results[0]
            
            # Check if test input has a similar pattern
            test_pattern = detect_pattern_repetition(test_input)
            
            if test_pattern:
                # Apply the output pattern scaling
                pattern, h_repeat, w_repeat = test_pattern
                target_h = test_input.shape[0] * (out_pattern[1] / in_pattern[1])
                target_w = test_input.shape[1] * (out_pattern[2] / in_pattern[2])
                
                if target_h.is_integer() and target_w.is_integer():
                    prediction = fill_pattern(int(target_h), int(target_w), pattern)
                    shape_predictions.append(prediction)
        
        # If we found any shape transformations, add them to predictions
        if shape_predictions:
            predictions.extend(shape_predictions[:2])
        else:
            # Default: return the input as prediction
            predictions.append(test_input.copy())
    
    # Ensure we have at least 2 predictions for each test input
    while len(predictions) < len(test_inputs) * 2:
        if len(predictions) > 0:
            predictions.append(predictions[-1].copy())
        else:
            predictions.append(test_inputs[0].copy())
    
    return predictions[:2]  # Return first 2 predictions

def object_manipulation_reasoning(task: Dict[str, Any]) -> List[np.ndarray]:
    """
    Apply object manipulation reasoning to solve a task.
    
    Args:
        task: Task data
        
    Returns:
        List of predicted output grids
    """
    # Extract input/output pairs from the task
    train_inputs = []
    train_outputs = []
    
    for pair in task.get('train', []):
        train_inputs.append(np.array(pair['input']))
        train_outputs.append(np.array(pair['output']))
    
    # Extract test inputs
    test_inputs = []
    test_data = task.get('test', [])
    
    if isinstance(test_data, dict):
        test_inputs.append(np.array(test_data['input']))
    else:
        for test in test_data:
            test_inputs.append(np.array(test['input']))
    
    # Generate predictions
    predictions = []
    
    for test_input in test_inputs:
        object_predictions = []
        
        # Find objects in test input
        test_objects = find_objects(test_input)
        
        # Try shift transformations
        shift_matches = []
        
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                if dr == 0 and dc == 0:
                    continue
                    
                matches = 0
                for i in range(len(train_inputs)):
                    shifted = shift_grid(train_inputs[i], dr, dc, wrap=True)
                    if shifted.shape == train_outputs[i].shape and np.array_equal(shifted, train_outputs[i]):
                        matches += 1
                
                if matches == len(train_inputs):
                    shift_matches.append((dr, dc))
        
        if shift_matches:
            dr, dc = shift_matches[0]
            prediction = shift_grid(test_input, dr, dc, wrap=True)
            object_predictions.append(prediction)
        
        # Try logical operations
        for op in ['and', 'or', 'xor']:
            matches = 0
            
            for i in range(len(train_inputs)):
                # Create a copy with 0/1 values for logical operations
                binary_input = (train_inputs[i] > 0).astype(int)
                
                # Try applying the operation with itself rotated
                for k in range(1, 4):
                    rotated = rotate_grid(binary_input, k)
                    if rotated.shape == binary_input.shape:
                        result = apply_logical_operation(binary_input, rotated, op)
                        
                        # Convert back to the original colors
                        color_map = {}
                        for r in range(binary_input.shape[0]):
                            for c in range(binary_input.shape[1]):
                                if binary_input[r, c] > 0:
                                    color_map[1] = binary_input[r, c]
                                    break
                            if 1 in color_map:
                                break
                        
                        if 1 in color_map:
                            result = apply_color_mapping(result, {1: color_map[1]})
                        
                        if result.shape == train_outputs[i].shape and np.array_equal(result, train_outputs[i]):
                            matches += 1
            
            if matches == len(train_inputs):
                # Apply the same logic to the test input
                binary_test = (test_input > 0).astype(int)
                rotated_test = rotate_grid(binary_test, 1)  # Use first rotation
                
                if rotated_test.shape == binary_test.shape:
                    result = apply_logical_operation(binary_test, rotated_test, op)
                    
                    # Convert back to original colors
                    color_map = {}
                    for r in range(binary_test.shape[0]):
                        for c in range(binary_test.shape[1]):
                            if binary_test[r, c] > 0:
                                color_map[1] = test_input[r, c]
                                break
                        if 1 in color_map:
                            break
                    
                    if 1 in color_map:
                        result = apply_color_mapping(result, {1: color_map[1]})
                    
                    object_predictions.append(result)
        
        # Try mathematical operations
        for op in ['add', 'subtract']:
            matches = 0
            
            for i in range(len(train_inputs)):
                # Try applying the operation with itself shifted
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        if dr == 0 and dc == 0:
                            continue
                            
                        shifted = shift_grid(train_inputs[i], dr, dc, wrap=False)
                        
                        # Only consider valid shapes
                        if shifted.shape == train_inputs[i].shape:
                            result = apply_mathematical_operation(train_inputs[i], shifted, op)
                            
                            if result.shape == train_outputs[i].shape and np.array_equal(result, train_outputs[i]):
                                matches += 1
            
            if matches == len(train_inputs):
                # Apply the same operation to the test input
                shifted_test = shift_grid(test_input, 1, 0, wrap=False)  # Use a simple shift
                
                if shifted_test.shape == test_input.shape:
                    result = apply_mathematical_operation(test_input, shifted_test, op)
                    object_predictions.append(result)
        
        # If we found any object manipulations, add them to predictions
        if object_predictions:
            predictions.extend(object_predictions[:2])
        else:
            # Default: return the input as prediction
            predictions.append(test_input.copy())
    
    # Ensure we have at least 2 predictions for each test input
    while len(predictions) < len(test_inputs) * 2:
        if len(predictions) > 0:
            predictions.append(predictions[-1].copy())
        else:
            predictions.append(test_inputs[0].copy())
    
    return predictions[:2]  # Return first 2 predictions

def rule_extraction_reasoning(task: Dict[str, Any]) -> List[np.ndarray]:
    """
    Apply rule extraction reasoning to solve a task.
    
    Args:
        task: Task data
        
    Returns:
        List of predicted output grids
    """
    # Extract input/output pairs from the task
    train_inputs = []
    train_outputs = []
    
    for pair in task.get('train', []):
        train_inputs.append(np.array(pair['input']))
        train_outputs.append(np.array(pair['output']))
    
    # Extract test inputs
    test_inputs = []
    test_data = task.get('test', [])
    
    if isinstance(test_data, dict):
        test_inputs.append(np.array(test_data['input']))
    else:
        for test in test_data:
            test_inputs.append(np.array(test['input']))
    
    # Generate predictions
    predictions = []
    
    for test_input in test_inputs:
        rule_predictions = []
        
        # Check if output is the same for all training examples
        if all(np.array_equal(train_outputs[0], output) for output in train_outputs[1:]):
            # Output is always the same regardless of input
            rule_predictions.append(train_outputs[0].copy())
        
        # Check if output is a fixed transformation of input
        transformations = [
            ("rotate_90", lambda g: rotate_grid(g, 1)),
            ("rotate_180", lambda g: rotate_grid(g, 2)),
            ("rotate_270", lambda g: rotate_grid(g, 3)),
            ("flip_horizontal", lambda g: flip_grid(g, 1)),
            ("flip_vertical", lambda g: flip_grid(g, 0)),
            ("transpose", transpose_grid),
            ("invert_colors", lambda g: 9 - g)
        ]
        
        for name, transform_fn in transformations:
            matches = 0
            
            for i in range(len(train_inputs)):
                transformed = transform_fn(train_inputs[i])
                if transformed.shape == train_outputs[i].shape and np.array_equal(transformed, train_outputs[i]):
                    matches += 1
            
            if matches == len(train_inputs):
                prediction = transform_fn(test_input)
                rule_predictions.append(prediction)
        
        # Check if output is a subset or expansion of input
        subset_matches = 0
        expansion_matches = 0
        
        for i in range(len(train_inputs)):
            # Check if output is a subset (cropped version) of input
            is_subset = False
            
            for r in range(train_inputs[i].shape[0] - train_outputs[i].shape[0] + 1):
                for c in range(train_inputs[i].shape[1] - train_outputs[i].shape[1] + 1):
                    cropped = crop_grid(train_inputs[i], r, c, train_outputs[i].shape[0], train_outputs[i].shape[1])
                    if np.array_equal(cropped, train_outputs[i]):
                        is_subset = True
                        break
                if is_subset:
                    break
            
            if is_subset:
                subset_matches += 1
            
            # Check if output is an expansion (padded version) of input
            if train_outputs[i].shape[0] >= train_inputs[i].shape[0] and train_outputs[i].shape[1] >= train_inputs[i].shape[1]:
                is_expansion = False
                
                for r in range(train_outputs[i].shape[0] - train_inputs[i].shape[0] + 1):
                    for c in range(train_outputs[i].shape[1] - train_inputs[i].shape[1] + 1):
                        region = train_outputs[i][r:r+train_inputs[i].shape[0], c:c+train_inputs[i].shape[1]]
                        if np.array_equal(region, train_inputs[i]):
                            is_expansion = True
                            # Compute padding values
                            pad_top = r
                            pad_left = c
                            pad_bottom = train_outputs[i].shape[0] - r - train_inputs[i].shape[0]
                            pad_right = train_outputs[i].shape[1] - c - train_inputs[i].shape[1]
                            break
                    if is_expansion:
                        break
                
                if is_expansion:
                    expansion_matches += 1
        
        # Apply subset rule if consistent
        if subset_matches == len(train_inputs):
            # Use center crop as default
            r = test_input.shape[0] // 4
            c = test_input.shape[1] // 4
            h = test_input.shape[0] // 2
            w = test_input.shape[1] // 2
            
            prediction = crop_grid(test_input, r, c, h, w)
            rule_predictions.append(prediction)
        
        # Apply expansion rule if consistent
        if expansion_matches == len(train_inputs):
            # Use padding as default
            padding = 1
            prediction = pad_grid(test_input, padding)
            rule_predictions.append(prediction)
        
        # If we found any rules, add them to predictions
        if rule_predictions:
            predictions.extend(rule_predictions[:2])
        else:
            # Default: return the input as prediction
            predictions.append(test_input.copy())
    
    # Ensure we have at least 2 predictions for each test input
    while len(predictions) < len(test_inputs) * 2:
        if len(predictions) > 0:
            predictions.append(predictions[-1].copy())
        else:
            predictions.append(test_inputs[0].copy())
    
    return predictions[:2]  # Return first 2 predictions

def solve_task(task: Dict[str, Any], methods: List[str] = None) -> List[np.ndarray]:
    """
    Solve an ARC task using specified reasoning methods.
    
    Args:
        task: Task data
        methods: List of reasoning methods to try
        
    Returns:
        List of predicted output grids (2 attempts)
    """
    if methods is None:
        methods = ["Pattern Matching", "Color Transformation"]
    
    method_solvers = {
        "Pattern Matching": pattern_matching_reasoning,
        "Color Transformation": color_transformation_reasoning,
        "Shape Detection": shape_detection_reasoning,
        "Object Manipulation": object_manipulation_reasoning,
        "Rule Extraction": rule_extraction_reasoning
    }
    
    all_predictions = []
    
    for method in methods:
        if method in method_solvers:
            solver = method_solvers[method]
            predictions = solver(task)
            all_predictions.extend(predictions)
    
    # If no predictions were generated, use a default prediction
    if not all_predictions:
        test_data = task.get('test', [])
        
        if isinstance(test_data, dict):
            default_input = np.array(test_data['input'])
        else:
            default_input = np.array(test_data[0]['input'])
        
        all_predictions = [default_input.copy(), default_input.copy()]
    
    # Ensure we return exactly 2 predictions
    if len(all_predictions) < 2:
        all_predictions.append(all_predictions[0].copy())
    
    return all_predictions[:2]

def evaluate_performance(task: Dict[str, Any], ground_truth: List[np.ndarray], 
                         methods: List[str] = None) -> Tuple[bool, float]:
    """
    Evaluate the performance of the solver on a task.
    
    Args:
        task: Task data
        ground_truth: List of ground truth output grids
        methods: List of reasoning methods to try
        
    Returns:
        Tuple of (correct_prediction, score)
    """
    predictions = solve_task(task, methods)
    
    correct = False
    score = 0.0
    
    for prediction in predictions:
        for gt in ground_truth:
            if np.array_equal(prediction, gt):
                correct = True
                score = 1.0
                break
        if correct:
            break
    
    return correct, score
