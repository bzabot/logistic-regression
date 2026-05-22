# Machine Learning I - Practical Assignment

This repository contains the practical assignment for Machine Learning I. The project studies how a from-scratch NumPy implementation of binary Logistic Regression behaves on imbalanced classification datasets, and compares the default model with several add-ons designed to improve minority-class detection.

The applied motivation is a fintech Research Intelligence setting: a RAG/Weaviate system can retrieve many candidate documents for a Wealth Manager query, but only a small number are truly relevant. This is analogous to binary class imbalance, where the positive/minority class is rare but important.

## Main Deliverables

- `src/analyze_addon_performance_results.ipynb`  
  Final notebook with the analysis, plots, explanations, and conclusions.

- `Presentation_BrunoZabot.pdf`  
  Final presentation used to defend the project.

- `presentation_roteiro_slides.md`  
  Detailed slide-by-slide speaking script used to prepare the presentation.

## Main Source Files

- `src/logistic_regression.py`  
  From-scratch Logistic Regression implementation using NumPy. It includes the default model and the add-ons:
  - `WeightedErrors`
  - `Gaar`
  - `CAGD`
  - `F1Threshold`

- `src/analyze_addon_performance.py`  
  Experiment runner. It trains all model variants across datasets and seeds, then writes result CSVs.

- `src/datasets.py`  
  Dataset loading and preprocessing utilities. It normalizes the target convention so that:
  - `target = 0` is the majority class
  - `target = 1` is the minority class

- `src/metrics.py`  
  Metrics for imbalanced binary classification, including balanced accuracy, minority recall, minority F1, confusion matrix components, MCC, ROC AUC, and average precision.

## Results and Outputs

The final analysis reads the stored experiment results from:

- `outputs/addon_performance/raw_results.csv`
- `outputs/addon_performance/model_summary.csv`
- `outputs/addon_performance/dataset_summary.csv`
- `outputs/addon_performance/config.json`

The full experiment contains:

- 19 datasets
- 20 random seeds
- 16 model variants
- 6080 successful training runs

By default, the notebook does not recompute all runs. It reads the saved CSV files above.

## How to Run

Install dependencies:

```bash
uv sync
```

Register the Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name assignment --display-name "assignment"
```

Open:

```text
src/analyze_addon_performance_results.ipynb
```

To reproduce the plots and tables from the stored results, keep:

```python
RUN_FULL_EXPERIMENT = False
```

To recompute the full experiment from scratch, set:

```python
RUN_FULL_EXPERIMENT = True
```

or run:

```bash
uv run python src/analyze_addon_performance.py
```

## Project Summary

The default Logistic Regression model uses a symmetric loss and a fixed threshold of 0.5. In highly imbalanced datasets, this tends to favor the majority class. The project evaluates whether modifications to the training process or the final decision threshold can improve minority-class detection.

The main findings are:

- `cagd_f1_threshold` achieved the best mean balanced accuracy.
- `gaar_cagd` was the most consistent model by average rank.
- `F1Threshold` improved minority recall and minority F1, but can increase false positives.
- `WeightedErrors` alone did not clearly improve the default model in this setup.

The main trade-off is between recovering more minority-class examples and preserving majority-class performance.

## Notes

The raw dataset directory `class_imbalance/` is ignored by Git because it is data-heavy. The notebook expects the datasets to be available locally when recomputing the full experiment.

