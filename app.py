import streamlit as st
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_loader import load_tasks, load_solutions, save_submission
from utils.visualizer import display_grid, display_task
from reasoning.solver import solve_task, evaluate_performance

# Set page config
st.set_page_config(
    page_title="ARC Prize Solver",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("Abstraction and Reasoning Corpus (ARC) Prize Solver")
st.markdown("""
This application helps solve ARC Prize tasks by discovering patterns in demonstration pairs and applying them to test inputs.
Explore the datasets, visualize tasks, and test different reasoning approaches.
""")

# Create the .streamlit directory if it doesn't exist
if not os.path.exists('.streamlit'):
    os.makedirs('.streamlit')

# Create config.toml if it doesn't exist
if not os.path.exists('.streamlit/config.toml'):
    with open('.streamlit/config.toml', 'w') as f:
        f.write("""
[server]
headless = true
address = "0.0.0.0"
port = 5000
""")

# Sidebar for dataset selection and options
st.sidebar.header("Dataset Options")

dataset_options = {
    "Sample Task": "sample_task.json"
}

# Uncomment these lines when you have the actual competition data files
# dataset_options.update({
#     "Training": "arc-agi_training-challenges.json",
#     "Evaluation": "arc-agi_evaluation-challenges.json",
#     "Test": "arc-agi_test-challenges.json"
# })

selected_dataset = st.sidebar.selectbox(
    "Select Dataset:",
    list(dataset_options.keys())
)

# Load dataset
try:
    dataset_path = dataset_options[selected_dataset]
    tasks = load_tasks(dataset_path)
    st.sidebar.success(f"Loaded {len(tasks)} tasks from {selected_dataset} dataset")
    
    # Load corresponding solutions if available (except for test dataset)
    solutions = None
    if selected_dataset in ["Sample Task", "Training", "Evaluation"]:
        if selected_dataset == "Sample Task":
            solution_path = "sample_solution.json"
        else:
            solution_path = dataset_path.replace("challenges", "solutions")
        
        try:
            solutions = load_solutions(solution_path)
            st.sidebar.success(f"Loaded solutions for {selected_dataset} dataset")
        except Exception as e:
            st.sidebar.warning(f"Couldn't load solutions: {str(e)}")
    
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    st.warning("Please ensure the dataset files are in the 'data' directory.")
    tasks = {}

# Solver options
st.sidebar.header("Solver Options")

reasoning_methods = [
    "Pattern Matching", 
    "Color Transformation", 
    "Shape Detection", 
    "Object Manipulation", 
    "Rule Extraction"
]

selected_methods = st.sidebar.multiselect(
    "Select Reasoning Methods:",
    reasoning_methods,
    default=["Pattern Matching", "Color Transformation"]
)

# Main content
tabs = st.tabs(["Task Explorer", "Solve Tasks", "Submission Generator", "Performance Analysis"])

with tabs[0]:
    st.header("Task Explorer")
    
    if tasks:
        task_ids = list(tasks.keys())
        selected_task_id = st.selectbox("Select a task:", task_ids)
        
        if selected_task_id:
            selected_task = tasks[selected_task_id]
            st.subheader(f"Task ID: {selected_task_id}")
            
            display_task(selected_task)
            
            # Display solution if available
            if solutions and selected_task_id in solutions:
                st.subheader("Ground Truth Output(s):")
                for output_idx, test_output in enumerate(solutions[selected_task_id]):
                    st.write(f"Test Output {output_idx + 1}:")
                    display_grid(test_output)
    else:
        st.info("No tasks loaded. Please select a dataset or check if the dataset files are in the 'data' directory.")

with tabs[1]:
    st.header("Solve Tasks")
    
    if tasks:
        solve_task_id = st.selectbox("Select a task to solve:", list(tasks.keys()), key="solve_task_select")
        
        if solve_task_id:
            task_to_solve = tasks[solve_task_id]
            
            st.subheader("Task to Solve:")
            display_task(task_to_solve)
            
            solve_button = st.button("Solve Task")
            
            if solve_button:
                with st.spinner("Solving task..."):
                    attempts = solve_task(task_to_solve, selected_methods)
                    
                    st.subheader("Prediction Attempts:")
                    for i, attempt in enumerate(attempts):
                        st.write(f"Attempt {i+1}:")
                        display_grid(attempt)
                    
                    # Check against ground truth if available
                    if solutions and solve_task_id in solutions:
                        ground_truth = solutions[solve_task_id]
                        correct = False
                        
                        for attempt in attempts:
                            for gt in ground_truth:
                                if np.array_equal(attempt, gt):
                                    correct = True
                                    break
                            if correct:
                                break
                        
                        if correct:
                            st.success("Task solved correctly! At least one attempt matches the ground truth.")
                        else:
                            st.error("None of the attempts match the ground truth.")
    else:
        st.info("No tasks loaded. Please select a dataset.")

with tabs[2]:
    st.header("Submission Generator")
    
    if tasks and selected_dataset == "Test":
        st.write("Generate a submission file for the test dataset in the ARC Prize format.")
        
        # Option to limit number of tasks to process
        process_limit = st.slider(
            "Number of tasks to process:",
            min_value=1,
            max_value=min(240, len(tasks)),
            value=min(10, len(tasks))
        )
        
        generate_button = st.button("Generate Submission")
        
        if generate_button:
            with st.spinner(f"Generating predictions for {process_limit} test tasks..."):
                submission = {}
                
                progress_bar = st.progress(0)
                
                for i, (task_id, task) in enumerate(list(tasks.items())[:process_limit]):
                    attempts = solve_task(task, selected_methods)
                    
                    # Format according to submission requirements for ARC Prize
                    task_submission = []
                    test_inputs = task['test']
                    
                    # If test is a dict, convert to list for consistent processing
                    if isinstance(test_inputs, dict):
                        test_inputs = [test_inputs]
                    
                    for test_idx in range(len(test_inputs)):
                        task_submission.append({
                            "attempt_1": attempts[0].tolist() if len(attempts) > 0 else [[0, 0]],
                            "attempt_2": attempts[1].tolist() if len(attempts) > 1 else [[0, 0]]
                        })
                    
                    submission[task_id] = task_submission
                    progress_bar.progress((i + 1) / process_limit)
                
                st.success(f"Generated predictions for {process_limit} tasks")
                
                # Display sample of the submission
                st.subheader("Sample of submission.json:")
                sample_keys = list(submission.keys())[:min(3, len(submission))]
                st.json({k: submission[k] for k in sample_keys})
                
                # Save submission to file
                save_path = os.path.join("data", "submission.json")
                save_submission(submission, save_path)
                
                # Option to download the submission
                with open(save_path, 'r') as f:
                    submission_json = f.read()
                
                st.download_button(
                    label="Download submission.json",
                    data=submission_json,
                    file_name="submission.json",
                    mime="application/json"
                )
                
                st.success(f"Submission file saved to {save_path}")
    else:
        st.info("Please select the Test dataset to generate a submission for the ARC Prize competition.")

with tabs[3]:
    st.header("Performance Analysis")
    
    if tasks and selected_dataset in ["Sample Task", "Training", "Evaluation"] and solutions:
        st.write("Analyze the performance of different reasoning methods on the dataset.")
        
        # Option to limit number of tasks to analyze
        analysis_limit = st.slider(
            "Number of tasks to analyze:",
            min_value=1,
            max_value=min(100, len(tasks)),
            value=min(20, len(tasks))
        )
        
        analyze_button = st.button("Analyze Performance")
        
        if analyze_button:
            with st.spinner(f"Analyzing performance on {analysis_limit} tasks..."):
                progress_bar = st.progress(0)
                
                performance_data = []
                
                for i, (task_id, task) in enumerate(list(tasks.items())[:analysis_limit]):
                    ground_truth = solutions.get(task_id)
                    
                    if ground_truth:
                        # Evaluate each method individually
                        for method in reasoning_methods:
                            correct, score = evaluate_performance(task, ground_truth, [method])
                            performance_data.append({
                                "Task ID": task_id,
                                "Method": method,
                                "Correct": correct,
                                "Score": score
                            })
                    
                    progress_bar.progress((i + 1) / analysis_limit)
                
                if performance_data:
                    # Create performance dataframe
                    performance_df = pd.DataFrame(performance_data)
                    
                    # Display performance metrics
                    st.subheader("Overall Performance")
                    
                    # Overall accuracy
                    st.metric("Overall Accuracy", f"{performance_df['Correct'].mean():.2%}")
                    
                    # Method performance comparison
                    st.subheader("Method Performance")
                    method_performance = performance_df.groupby("Method")["Correct"].mean().reset_index()
                    method_performance["Accuracy"] = method_performance["Correct"].apply(lambda x: f"{x:.2%}")
                    
                    st.dataframe(method_performance[["Method", "Accuracy"]])
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(method_performance["Method"], method_performance["Correct"])
                    ax.set_xlabel("Reasoning Method")
                    ax.set_ylabel("Accuracy")
                    ax.set_title("Performance by Reasoning Method")
                    ax.set_ylim(0, 1)
                    plt.xticks(rotation=45, ha="right")
                    
                    st.pyplot(fig)
                    
                    # Task-level performance
                    st.subheader("Task-level Performance")
                    task_performance = performance_df.groupby("Task ID")["Correct"].max().reset_index()
                    task_performance.columns = ["Task ID", "Solved"]
                    
                    st.dataframe(task_performance.sort_values("Solved", ascending=False))
                else:
                    st.warning("No performance data collected. This could be because no ground truth solutions were found for the selected tasks.")
    else:
        st.info("Please select Sample Task, Training, or Evaluation dataset with solutions for performance analysis.")

# Footer
st.markdown("---")
st.markdown("ARC Prize Solver - Abstraction and Reasoning Corpus Challenge")

# Add repository link
st.sidebar.markdown("---")
st.sidebar.markdown("[GitHub Repository](https://github.com/iamabhinav13/arc-solver)")
st.sidebar.markdown("---")
st.sidebar.markdown("ARC Prize Competition 2025")
