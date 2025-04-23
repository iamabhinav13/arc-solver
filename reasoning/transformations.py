import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
import copy

def apply_color_mapping(grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    """
    Apply a color mapping to a grid.
    
    Args:
        grid: Input grid
        mapping: Dictionary mapping input colors to output colors
        
    Returns:
        Transformed grid
    """
    result = np.zeros_like(grid)
    
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            color = grid[i, j]
            result[i, j] = mapping.get(color, color)  # Use original if no mapping
    
    return result

def rotate_grid(grid: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Rotate a grid 90 degrees k times counterclockwise.
    
    Args:
        grid: Input grid
        k: Number of 90-degree rotations
        
    Returns:
        Rotated grid
    """
    return np.rot90(grid, k)

def flip_grid(grid: np.ndarray, axis: int) -> np.ndarray:
    """
    Flip a grid along an axis.
    
    Args:
        grid: Input grid
        axis: 0 for vertical flip, 1 for horizontal flip
        
    Returns:
        Flipped grid
    """
    return np.flip(grid, axis)

def scale_grid(grid: np.ndarray, factor: int) -> np.ndarray:
    """
    Scale a grid by repeating each cell.
    
    Args:
        grid: Input grid
        factor: Scaling factor
        
    Returns:
        Scaled grid
    """
    h, w = grid.shape
    result = np.zeros((h * factor, w * factor), dtype=grid.dtype)
    
    for i in range(h):
        for j in range(w):
            result[i*factor:(i+1)*factor, j*factor:(j+1)*factor] = grid[i, j]
    
    return result

def crop_grid(grid: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
    """
    Crop a section from a grid.
    
    Args:
        grid: Input grid
        top, left: Top-left corner coordinates
        height, width: Dimensions of the crop
        
    Returns:
        Cropped grid
    """
    return grid[top:top+height, left:left+width]

def pad_grid(grid: np.ndarray, padding: int, value: int = 0) -> np.ndarray:
    """
    Pad a grid with a constant value.
    
    Args:
        grid: Input grid
        padding: Number of cells to pad on all sides
        value: Value to use for padding
        
    Returns:
        Padded grid
    """
    return np.pad(grid, padding, constant_values=value)

def replace_color(grid: np.ndarray, old_color: int, new_color: int) -> np.ndarray:
    """
    Replace a specific color in a grid.
    
    Args:
        grid: Input grid
        old_color: Color to replace
        new_color: New color
        
    Returns:
        Grid with replaced colors
    """
    result = grid.copy()
    result[result == old_color] = new_color
    return result

def apply_logical_operation(grid1: np.ndarray, grid2: np.ndarray, operation: str) -> np.ndarray:
    """
    Apply a logical operation between two grids.
    
    Args:
        grid1, grid2: Input grids
        operation: 'and', 'or', 'xor', or 'not' (for grid1 only)
        
    Returns:
        Result of the operation
    """
    if grid1.shape != grid2.shape and operation != 'not':
        raise ValueError("Grids must have the same shape for logical operations")
    
    if operation == 'and':
        return np.logical_and(grid1, grid2).astype(int)
    elif operation == 'or':
        return np.logical_or(grid1, grid2).astype(int)
    elif operation == 'xor':
        return np.logical_xor(grid1, grid2).astype(int)
    elif operation == 'not':
        return np.logical_not(grid1).astype(int)
    else:
        raise ValueError(f"Unsupported operation: {operation}")

def apply_mathematical_operation(grid1: np.ndarray, grid2: np.ndarray, operation: str, mod: int = 10) -> np.ndarray:
    """
    Apply a mathematical operation between two grids.
    
    Args:
        grid1, grid2: Input grids
        operation: 'add', 'subtract', 'multiply', or 'mod' (modular addition)
        mod: Modulus for modular operations
        
    Returns:
        Result of the operation
    """
    if grid1.shape != grid2.shape:
        raise ValueError("Grids must have the same shape for mathematical operations")
    
    if operation == 'add':
        return (grid1 + grid2) % mod
    elif operation == 'subtract':
        return (grid1 - grid2) % mod
    elif operation == 'multiply':
        return (grid1 * grid2) % mod
    elif operation == 'mod':
        # Modular addition
        return (grid1 + grid2) % mod
    else:
        raise ValueError(f"Unsupported operation: {operation}")

def shift_grid(grid: np.ndarray, shift_r: int, shift_c: int, wrap: bool = False) -> np.ndarray:
    """
    Shift a grid in a specified direction.
    
    Args:
        grid: Input grid
        shift_r: Row shift (positive for down, negative for up)
        shift_c: Column shift (positive for right, negative for left)
        wrap: Whether to wrap around (True) or fill with zeros (False)
        
    Returns:
        Shifted grid
    """
    h, w = grid.shape
    result = np.zeros_like(grid)
    
    for i in range(h):
        for j in range(w):
            new_i = (i + shift_r) % h if wrap else i + shift_r
            new_j = (j + shift_c) % w if wrap else j + shift_c
            
            if 0 <= new_i < h and 0 <= new_j < w:
                result[new_i, new_j] = grid[i, j]
    
    return result

def transpose_grid(grid: np.ndarray) -> np.ndarray:
    """
    Transpose a grid (swap rows and columns).
    
    Args:
        grid: Input grid
        
    Returns:
        Transposed grid
    """
    return grid.T

def fill_pattern(height: int, width: int, pattern: np.ndarray) -> np.ndarray:
    """
    Fill a grid of specified dimensions with a repeating pattern.
    
    Args:
        height, width: Dimensions of the output grid
        pattern: Pattern to repeat
        
    Returns:
        Grid filled with the pattern
    """
    pattern_h, pattern_w = pattern.shape
    result = np.zeros((height, width), dtype=pattern.dtype)
    
    for i in range(0, height, pattern_h):
        for j in range(0, width, pattern_w):
            # Calculate the portion of the pattern that fits
            h = min(pattern_h, height - i)
            w = min(pattern_w, width - j)
            result[i:i+h, j:j+w] = pattern[:h, :w]
    
    return result

def get_transformation_function(transform_name: str) -> Callable:
    """
    Get a transformation function by name.
    
    Args:
        transform_name: Name of the transformation
        
    Returns:
        Transformation function
    """
    transformations = {
        "rotate_90": lambda grid: rotate_grid(grid, 1),
        "rotate_180": lambda grid: rotate_grid(grid, 2),
        "rotate_270": lambda grid: rotate_grid(grid, 3),
        "flip_horizontal": lambda grid: flip_grid(grid, 1),
        "flip_vertical": lambda grid: flip_grid(grid, 0),
        "transpose": transpose_grid,
        "scale_2x": lambda grid: scale_grid(grid, 2),
        "invert_colors": lambda grid: 9 - grid
    }
    
    return transformations.get(transform_name)
