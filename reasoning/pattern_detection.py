import numpy as np
from typing import Dict, List, Tuple, Optional, Set

def extract_colors(grid: np.ndarray) -> Set[int]:
    """
    Extract the set of colors (integers) used in a grid.
    
    Args:
        grid: 2D numpy array
        
    Returns:
        Set of unique integers in the grid
    """
    return set(np.unique(grid))

def count_color_occurrences(grid: np.ndarray) -> Dict[int, int]:
    """
    Count occurrences of each color in a grid.
    
    Args:
        grid: 2D numpy array
        
    Returns:
        Dictionary mapping colors to their counts
    """
    unique, counts = np.unique(grid, return_counts=True)
    return dict(zip(unique, counts))

def detect_color_mapping(input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[int, int]:
    """
    Detect if there's a consistent color mapping from input to output.
    
    Args:
        input_grid: Input grid
        output_grid: Output grid
        
    Returns:
        Dictionary mapping input colors to output colors, or empty dict if no consistent mapping
    """
    if input_grid.shape != output_grid.shape:
        return {}
    
    mapping = {}
    
    for i in range(input_grid.shape[0]):
        for j in range(input_grid.shape[1]):
            in_color = input_grid[i, j]
            out_color = output_grid[i, j]
            
            if in_color in mapping:
                if mapping[in_color] != out_color:
                    # Inconsistent mapping
                    return {}
            else:
                mapping[in_color] = out_color
    
    return mapping

def find_objects(grid: np.ndarray, background: int = 0) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Find distinct objects in a grid, separated by background color.
    
    Args:
        grid: 2D numpy array
        background: Background color (defaults to 0)
        
    Returns:
        List of (object_grid, position) tuples where position is the top-left corner
    """
    objects = []
    visited = np.zeros_like(grid, dtype=bool)
    
    def dfs(i, j, object_pixels):
        if (i < 0 or i >= grid.shape[0] or 
            j < 0 or j >= grid.shape[1] or 
            visited[i, j] or grid[i, j] == background):
            return
        
        visited[i, j] = True
        object_pixels.append((i, j))
        
        # Check all 4 directions
        dfs(i+1, j, object_pixels)
        dfs(i-1, j, object_pixels)
        dfs(i, j+1, object_pixels)
        dfs(i, j-1, object_pixels)
    
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not visited[i, j] and grid[i, j] != background:
                object_pixels = []
                dfs(i, j, object_pixels)
                
                if object_pixels:
                    # Find bounding box
                    min_i = min(p[0] for p in object_pixels)
                    max_i = max(p[0] for p in object_pixels)
                    min_j = min(p[1] for p in object_pixels)
                    max_j = max(p[1] for p in object_pixels)
                    
                    # Extract object grid
                    obj_height = max_i - min_i + 1
                    obj_width = max_j - min_j + 1
                    obj_grid = np.full((obj_height, obj_width), background)
                    
                    for pi, pj in object_pixels:
                        obj_grid[pi - min_i, pj - min_j] = grid[pi, pj]
                    
                    objects.append((obj_grid, (min_i, min_j)))
    
    return objects

def detect_symmetry(grid: np.ndarray) -> Dict[str, bool]:
    """
    Detect various types of symmetry in a grid.
    
    Args:
        grid: 2D numpy array
        
    Returns:
        Dictionary with symmetry types and boolean values
    """
    h, w = grid.shape
    
    # Check horizontal symmetry
    horizontal = True
    for i in range(h):
        for j in range(w // 2):
            if grid[i, j] != grid[i, w - j - 1]:
                horizontal = False
                break
        if not horizontal:
            break
    
    # Check vertical symmetry
    vertical = True
    for j in range(w):
        for i in range(h // 2):
            if grid[i, j] != grid[h - i - 1, j]:
                vertical = False
                break
        if not vertical:
            break
    
    # Check diagonal symmetry (main diagonal)
    diagonal1 = False
    if h == w:  # Only square matrices can have diagonal symmetry
        diagonal1 = True
        for i in range(h):
            for j in range(i):
                if grid[i, j] != grid[j, i]:
                    diagonal1 = False
                    break
            if not diagonal1:
                break
    
    # Check diagonal symmetry (anti-diagonal)
    diagonal2 = False
    if h == w:
        diagonal2 = True
        for i in range(h):
            for j in range(w):
                if i + j != h - 1:
                    continue
                if grid[i, j] != grid[h - j - 1, w - i - 1]:
                    diagonal2 = False
                    break
            if not diagonal2:
                break
    
    return {
        'horizontal': horizontal,
        'vertical': vertical,
        'diagonal1': diagonal1,
        'diagonal2': diagonal2
    }

def detect_pattern_repetition(grid: np.ndarray) -> Optional[Tuple[np.ndarray, int, int]]:
    """
    Detect if a grid consists of a smaller pattern repeated.
    
    Args:
        grid: 2D numpy array
        
    Returns:
        Tuple of (pattern, rows_repeat, cols_repeat) or None if no pattern found
    """
    h, w = grid.shape
    
    # Try different pattern sizes
    for pattern_h in range(1, h + 1):
        if h % pattern_h != 0:
            continue
            
        for pattern_w in range(1, w + 1):
            if w % pattern_w != 0:
                continue
                
            # Extract the potential pattern
            pattern = grid[:pattern_h, :pattern_w]
            
            # Check if it repeats throughout the grid
            is_repeating = True
            
            for i in range(0, h, pattern_h):
                for j in range(0, w, pattern_w):
                    if not np.array_equal(grid[i:i+pattern_h, j:j+pattern_w], pattern):
                        is_repeating = False
                        break
                if not is_repeating:
                    break
            
            if is_repeating and (pattern_h < h or pattern_w < w):
                return pattern, h // pattern_h, w // pattern_w
    
    return None

def detect_size_changes(input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, float]:
    """
    Detect how the size changes from input to output.
    
    Args:
        input_grid: Input grid
        output_grid: Output grid
        
    Returns:
        Dictionary with scaling factors
    """
    input_h, input_w = input_grid.shape
    output_h, output_w = output_grid.shape
    
    return {
        'height_scale': output_h / input_h,
        'width_scale': output_w / input_w,
        'area_scale': (output_h * output_w) / (input_h * input_w)
    }
