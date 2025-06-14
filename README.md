# Exploring Language Model Scaling Laws Under Learning Rate Schedules

This is the final project for the "Topics in Deep Learning Theory" course at Peking University, aiming to reproduce and explore the impact of Learning Rate Schedules (LRS) on the scaling laws during the pretraining of Large Language Models (LLMs).

The project is primarily based on the following two core papers:

* Tissue, H., et al. (2024). *Scaling law with learning rate annealing*. arXiv:2408.11029. (Momentum Law)
* Luo, K., et al. (2024). *A multi-power law for loss curve prediction across learning rate schedules*. (Multi-Power Law)

## File Structure

This repository contains the following three core Python scripts:

* `scaling_law_experiments.py`: For reproducing the core scaling laws (Momentum Law and Multi-Power Law) from the papers mentioned above.
* `advanced_scaling_exploration.py`: Contains our team's original exploratory work, including proposing a new Hybrid Scaling Law, theoretical analysis, a joint fitting strategy, and an automatic optimal scheduler design.
* `advanced_analysis_report.py`: For summarizing the results of the exploratory experiments and generating comprehensive visualizations and detailed analysis reports.

## Environment Dependencies

This project requires the following Python libraries:

* `pandas`
* `numpy`
* `matplotlib`
* `scipy`
* `tqdm`
* `seaborn`

You can install them via pip:
```bash
pip install pandas numpy matplotlib scipy tqdm seaborn
```

## Data

All experiments are based on the `gpt_loss+lrs.pkl` data file provided for the course. This file contains the loss (`Metrics/loss`) and learning rate (`lr`) data from training a 100M-parameter GPT model on 20B tokens, using three different learning rate schedules (`811`, `wsd`, `cosine`).

**Please ensure the `gpt_loss+lrs.pkl` file is in the same directory as all Python scripts.**

## How to Run

### 1. Reproduce Experiments

To run the reproduction experiments for the Tissue and Luo laws, execute:

```bash
python scaling_law_experiments.py
```

This script will:
1.  Load the `gpt_loss+lrs.pkl` data.
2.  Use the `cosine` scheduler's loss curve to fit the parameters for both laws.
3.  Predict the models' performance on the `811` and `wsd` schedulers.
4.  Print the fitted parameters and $R^2$ evaluation metrics to the console.
5.  Generate two plots: `scaling_law_reproduction_results_clean.png` (loss curve comparison) and `model_comparison_summary_clean.png` ($R^2$ score comparison).

### 2. Run Original Exploratory Experiments

To run our team's further explorations, execute:

```bash
python advanced_scaling_exploration.py
```

This script will perform our four core explorations:
1.  **Hybrid Scaling Law**: Fits the model on `cosine` and evaluates its performance on `811` and `wsd`.
2.  **Theoretical Analysis of the $\alpha$ Exponent**: Empirically estimates the value of $\alpha$.
3.  **Joint Fitting Strategy**: Jointly fits the Tissue model on `cosine` and `wsd` and tests its generalization on `811`.
4.  **Optimal LRS Design**: Uses the Hybrid model as a surrogate to automatically design a more optimal LRS.

The script will output a detailed experimental process and results to the console.

### 3. Generate Summary Report and Plots for Explorations

To generate a summary visualization and text report for the exploratory work, execute:

```bash
python advanced_analysis_report.py
```

This script calls the core modules from `advanced_scaling_exploration.py` and generates:
* `advanced_scaling_exploration_comprehensive.png`: A comprehensive 8-panel plot systematically showcasing all exploratory findings.
* `advanced_scaling_exploration_report.txt`: A detailed text report providing in-depth interpretations of each finding.

## Summary of Core Findings

### Reproduction Section

* **Tissue's Momentum Law** demonstrated better **generalization capability** in our reproduction.
* **Luo's Multi-Power Law**, while more complex, showed a tendency to **overfit** to the training schedule in our setup, resulting in poorer generalization.

### Exploration Section

1.  **Hybrid Scaling Law**: We proposed a new $S_3$ term to capture sharp jumps in the learning rate. The model performed well on `wsd` but failed on `811` due to overfitting of the $S_3$ term's parameters, revealing its sensitivity to extreme, unseen cases.
2.  **New Insight on the $\alpha$ Exponent**: Through direct empirical fitting, we found that in our 100M model setup, the power-law exponent of loss decay with $S_1$ is $\alpha \approx 0.05$. This significantly **challenges the common theoretical paradigm of $\alpha \approx 3$** and suggests a regime-dependent nature for $\alpha$.
3.  **Joint Fitting Strategy**: By jointly fitting the Tissue model on multiple schedulers (`cosine` + `wsd`), its $R^2$ score on the unseen `811` schedule **significantly improved from 0.738 to 0.871**, proving that joint fitting is an effective method for enhancing generalization.
4.  **Automatic Optimal Scheduler Design**: Using our proposed Hybrid model as a surrogate, we successfully designed an optimized LRS with a predicted final loss lower than all baseline schedulers, validating the "scaling-law-as-a-controller" approach.
