import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional

# Color map for visualization
ARC_COLORS = {
    0: '#FFFFFF',  # White
    1: '#000000',  # Black
    2: '#0074D9',  # Blue
    3: '#FF4136',  # Red
    4: '#2ECC40',  # Green
    5: '#FFDC00',  # Yellow
    6: '#AAAAAA',  # Gray
    7: '#F012BE',  # Magenta
    8: '#FF851B',  # Orange
    9: '#7FDBFF',  # Light blue
}

def display_grid(grid: np.ndarray, ax: Optional[plt.Axes] = None, title: Optional[str] = None) -> plt.Axes:
    """
    Display a grid with the ARC color scheme.
    
    Args:
        grid: 2D numpy array containing integers from 0-9
        ax: Optional matplotlib axis to plot on
        title: Optional title for the plot
        
    Returns:
        The matplotlib axis object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    # Plot the grid
    height, width = grid.shape
    
    # Create a colored version of the grid
    colored_grid = np.zeros((height, width, 3))
    
    for i in range(height):
        for j in range(width):
            color = ARC_COLORS.get(grid[i, j], '#FFFFFF')
            
            # Convert hex color to RGB
            r = int(color[1:3], 16) / 255.0
            g = int(color[3:5], 16) / 255.0
            b = int(color[5:7], 16) / 255.0
            
            colored_grid[i, j] = [r, g, b]
    
    ax.imshow(colored_grid, aspect='equal')
    
    # Draw grid lines
    for i in range(width + 1):
        ax.axvline(i - 0.5, color='gray', linewidth=0.5)
    for i in range(height + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
    
    # Set title if provided
    if title:
        ax.set_title(title)
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    return ax

def display_task(task: Dict[str, Any]) -> None:
    """
    Display a complete ARC task in Streamlit.
    
    Args:
        task: Task data
    """
    # Split the display into train and test sections
    train_col, test_col = st.columns([2, 1])
    
    # Display training pairs
    with train_col:
        st.subheader("Training Pairs")
        
        for i, pair in enumerate(task.get('train', [])):
            pair_cols = st.columns(2)
            
            with pair_cols[0]:
                st.write(f"Pair {i+1} Input:")
                fig, ax = plt.subplots(figsize=(5, 5))
                display_grid(np.array(pair['input']), ax)
                st.pyplot(fig)
            
            with pair_cols[1]:
                st.write(f"Pair {i+1} Output:")
                fig, ax = plt.subplots(figsize=(5, 5))
                display_grid(np.array(pair['output']), ax)
                st.pyplot(fig)
    
    # Display test inputs
    with test_col:
        st.subheader("Test Input(s)")
        
        test_data = task.get('test', [])
        
        # Handle both single test and multiple tests
        if isinstance(test_data, dict):
            fig, ax = plt.subplots(figsize=(5, 5))
            display_grid(np.array(test_data['input']), ax, title="Test Input")
            st.pyplot(fig)
        else:
            for i, test in enumerate(test_data):
                st.write(f"Test Input {i+1}:")
                fig, ax = plt.subplots(figsize=(5, 5))
                display_grid(np.array(test['input']), ax)
                st.pyplot(fig)

def visualize_transformation(input_grid: np.ndarray, output_grid: np.ndarray, 
                            intermediate_steps: Optional[List[np.ndarray]] = None) -> None:
    """
    Visualize a transformation from input to output, optionally with intermediate steps.
    
    Args:
        input_grid: Input grid
        output_grid: Output grid
        intermediate_steps: Optional list of intermediate transformation steps
    """
    num_steps = 2 if intermediate_steps is None else 2 + len(intermediate_steps)
    
    fig, axes = plt.subplots(1, num_steps, figsize=(5 * num_steps, 5))
    
    # Display input
    display_grid(input_grid, axes[0], title="Input")
    
    # Display intermediate steps if provided
    if intermediate_steps:
        for i, step in enumerate(intermediate_steps):
            display_grid(step, axes[i+1], title=f"Step {i+1}")
    
    # Display output
    display_grid(output_grid, axes[-1], title="Output")
    
    st.pyplot(fig)
