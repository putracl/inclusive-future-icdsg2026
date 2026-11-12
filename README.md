# Inclusive Future: Strengths-Based Occupational Ranking Prototype

This repository contains the official code and reproducibility package for the paper:

**"Inclusive Future: A Proof of Concept for a Strengths-Based Occupational Ranking Prototype for Disability-Inclusive Career Guidance"** (Accepted at the 2026 2nd International Conference on Data Science and Geoinformatics - ICDSG 2026).

## Overview

This project implements a proof-of-concept occupational ranking prototype designed specifically for disabled students leaving Indonesian vocational secondary schools (SMK). Rather than steering candidates toward stereotypically "disability-appropriate" roles, this system evaluates candidate strengths against 10 O\*NET-anchored dimensions.

The prototype is strictly defined by **four responsible data commitments**:
1. **Disability is not an input:** To prevent algorithmic proxy discrimination and mitigate employment risk, disability status/type is never recorded as a modelling feature.
2. **Skill match and demand are separate signals:** Projected labour demand is reported alongside the skill match, never mathematically combined, ensuring candidates are not nudged toward roles unrelated to their actual strengths.
3. **Open reasoning:** Explanations are transparent, showing the per-dimension mathematical contributions driving each match.
4. **Documented mapping:** The O\*NET-to-dimension mapping is documented, explicit, and open to challenge.

## Repository Structure

All scripts and data are contained within the `Code/` directory:

* **`Inclusive Future.ipynb`**: An interactive Jupyter Notebook demonstrating the full end-to-end workflow—from vector construction to the final ranking and visualization. **(Recommended starting point)**
* **`compute_vectors.py`**: Constructs the occupation vectors by mapping O\*NET Skills Importance ratings and BLS 2024–2034 employment projections.
* **`recommender.py`**: Contains the core algorithmic logic, including the cosine similarity scoring and the separation of the demand signal context.
* **`candidate_b_deaf.py`**: Evaluates the synthetic Candidate B (a Deaf SMK graduate) under both hearing-default and Deaf-inclusive (Sunyi-style) scoring. 
* **`robustness_checks.py`**: Runs diagnostic tests, including the null baseline check against 1,000 random vectors.
* **`perturbation_check.py`**: Executes deterministic ±0.05 perturbation stability checks.
* **`bias_check.py`**: Conducts baseline comparisons (skill-only vs. demand-only rankings) and 12-profile sample coverage checks.
* **`figure1_contributions.py` & `figure1_data.csv`**: Data and plotting scripts to generate the stacked bar charts showing per-dimension contributions.

## How to Run the Code

You can explore this project in two ways: interactively via Google Colab or locally on your machine.

### Option 1: Google Colab (Easiest)
You can run the full pipeline directly in your browser without installing anything by opening the interactive notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/inclusive-future-icdsg2026/blob/main/Code/Inclusive%20Future.ipynb)
*(Note: Replace the URL above with your actual Colab share link or GitHub path once uploaded).*

### Option 2: Run Locally
The codebase is designed to be lightweight and auditable without heavy machine learning frameworks.

**Requirements:**
* Python 3.11+
* `numpy` (for vector manipulation)
* `matplotlib` (for generating visual figures)
* `jupyter` (optional, to run the notebook locally)

**Installation & Execution:**
```bash
# Install dependencies
pip install numpy matplotlib jupyter

# Run specific evaluations from the command line:
python Code/candidate_b_deaf.py
python Code/perturbation_check.py
python Code/robustness_checks.py
```

## Reproducibility Note

Executing the scripts provided here will precisely reproduce:
* Every numerical value reported in **Table II** (Candidate B modality scoring shifts) and **Table III** (Baseline comparisons).
* The stacked bar chart visuals (**Figure 1/2**).
* All quantitative metrics reported in the Abstract and Methodology sections (e.g., the 0.78–0.86 null baseline similarity, the ~0.14 headroom compression, and Spearman rank correlations). All stochastic checks are locked with `random.seed(42)`.
