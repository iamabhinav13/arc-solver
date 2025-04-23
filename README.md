# ARC Solver

A system to solve Abstraction and Reasoning Corpus (ARC) tasks by discovering patterns from demonstration pairs and applying them to test inputs.

## Project Overview

The Abstraction and Reasoning Corpus (ARC) is a benchmark measuring an AI system's ability to efficiently learn new skills. This project aims to develop a system that can:

1. Analyze patterns in demonstration input/output pairs
2. Apply discovered patterns to new test inputs
3. Provide accurate predictions for ARC tasks

## Features

- Task Explorer: Visualize tasks and their solutions
- Solve Tasks: Apply different reasoning methods to solve ARC tasks
- Submission Generator: Create submission files for the ARC competition
- Performance Analysis: Analyze the performance of different reasoning methods

## Reasoning Methods

- Pattern Matching: Identifying repeating patterns in grids
- Color Transformation: Mapping colors between input and output
- Shape Detection: Identifying and manipulating shapes
- Object Manipulation: Moving and transforming objects within grids
- Rule Extraction: Discovering underlying rules in transformations

## Technologies Used

- Python 3.11
- Streamlit for web interface
- NumPy for array manipulation
- Matplotlib for visualization

## Getting Started

To run this project:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

This project works with the ARC dataset, which consists of:
- Training challenges
- Evaluation challenges
- Test challenges

Each task contains demonstration pairs (input/output) and test inputs that need to be solved.