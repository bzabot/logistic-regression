# Improving `logistic_regression.py` for Highly Imbalanced Datasets

## Scope

This document studies the current NumPy implementation in [logistic_regression.py](/home/bruno-zabot/Studies/MachineLearning/assignment/logistic_regression.py) and proposes five concrete ways to improve logistic regression for highly imbalanced datasets in `class_imbalance/`.

The goal is not to replace logistic regression with a different model, but to make this implementation substantially more robust when the minority class is rare.

## What I observed in this project

### 1. The current implementation is a plain baseline

The current training loop:

- uses standard binary cross-entropy with no class weighting
- uses full-batch gradient descent with fixed `learning_rate=0.01`
- uses fixed `iterations=1000`
- does not regularize
- does not tune the classification threshold
- does not calibrate probabilities
- does not guard the sigmoid against overflow on large scores

For balanced data this is acceptable as a teaching baseline. For imbalanced data it is not enough.

### 2. The dataset folder contains several severe imbalance regimes

A few representative imbalance ratios from `class_imbalance/`:

- `dataset_1056_mc1.csv`: `9398 / 68` -> `138.21:1`
- `dataset_316_yeast_ml8.csv`: `2383 / 34` -> `70.09:1`
- `dataset_951_arsenic-male-lung.csv`: `546 / 13` -> `42.00:1`
- `dataset_1039_hiva_agnostic.csv`: `4080 / 149` -> `27.38:1`
- `dataset_311_oil_spill.csv`: `896 / 41` -> `21.85:1`

This is exactly the regime where standard logistic regression often learns a majority-friendly boundary and still reports deceptively reasonable accuracy.

### 3. The current model already shows the expected failure mode

On a quick holdout check using the current code and a default `0.5` threshold:

- `dataset_865_analcatdata_neavote.csv`: recall `0.000`
- `dataset_1013_analcatdata_challenger.csv`: recall `0.000`
- `dataset_311_oil_spill.csv`: recall `0.000`
- `dataset_1056_mc1.csv`: recall `0.000`

In addition, the current `sigmoid` overflowed on some runs because it computes `np.exp(-y)` directly with no stabilization.

## Why standard logistic regression struggles here

With heavy skew, standard maximum-likelihood logistic regression optimizes average log-loss across all samples. That objective is dominated by the majority class. As a result:

- gradients are driven mostly by majority examples
- the fitted intercept shifts toward the majority prior
- predicted probabilities for minority cases are often too small
- a fixed `0.5` decision threshold becomes inappropriate
- on very small rare-event datasets, coefficient and probability estimates can be biased

The five improvements below address these failure modes directly.

## Proposed Approaches

## 1. Cost-sensitive logistic regression with class-weighted cross-entropy

### What to change

Replace the unweighted loss with a weighted binary cross-entropy:

\[
\mathcal{L} = - \frac{1}{n}\sum_{i=1}^{n}\left[w_1 y_i \log(p_i) + w_0 (1-y_i)\log(1-p_i)\right]
\]

and propagate the same weights into the gradients.

Typical starting point:

- `w_c = n / (K * n_c)` for class `c`
- for binary tasks, minority class weight > majority class weight

### Why it helps

This is the most direct fix. It makes minority mistakes more expensive during optimization, so the model is no longer rewarded mainly for predicting the majority class well.

For this repository, it is the highest-impact first improvement because the current training loop is otherwise entirely majority-driven.

### How it maps to this codebase

Add optional parameters such as:

- `class_weight=None | "balanced" | {0: w0, 1: w1}`
- `sample_weight=None`

Then change the error term so each sample contributes proportionally to its weight before computing `dw` and `db`.

### Tradeoffs

- If weights are too aggressive, precision may fall.
- Weighted training changes the effective prior, so raw probabilities may need calibration afterward.

### Recommendation

Implement this first. It is simple, standard, and strongly supported by both practice and literature.

## 2. Rare-events logistic regression correction for intercept and probability bias

### What to change

Augment standard logistic regression with rare-events corrections inspired by King and Zeng:

- adjust estimation for rare-event bias
- correct the intercept when the event rate in the training sample differs from the real population rate
- support case-control style sampling if you later subsample the majority class

### Why it helps

King and Zeng showed that ordinary logistic regression can underestimate event probabilities in rare-event settings. This matters in your dataset folder because some tasks have extreme skew and small positive counts, where the intercept is especially vulnerable.

This is not the same as class weighting. Class weighting changes optimization pressure. Rare-events correction targets statistical bias in estimated probabilities and intercepts.

### How it maps to this codebase

Add an optional rare-events mode, for example:

- `rare_events_correction=False`
- optional `true_event_rate=None`

If the training set is artificially resampled, correct the fitted intercept back to the deployment prior. This is especially relevant if you combine this approach with under-sampling.

### Best fit in this repository

This approach is especially relevant for datasets like:

- `dataset_1056_mc1.csv`
- `dataset_316_yeast_ml8.csv`
- `dataset_311_oil_spill.csv`
- `dataset_1013_analcatdata_challenger.csv`

### Tradeoffs

- More statistical machinery than plain weighting
- Most useful when the minority event is genuinely rare or when training data is sampled differently from deployment data

### Recommendation

Implement this after weighted loss if probability quality matters, not just classification labels.

## 3. Focal logistic loss or dynamic hard-example reweighting

### What to change

Replace plain BCE with a focal-style loss:

\[
\mathrm{FL}(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)
\]

where:

- `alpha_t` handles class imbalance
- `gamma > 0` down-weights easy examples and focuses training on hard or misclassified ones

### Why it helps

In heavy imbalance, most majority examples become easy very quickly. Standard BCE keeps spending gradient budget on them. Focal loss suppresses those easy examples and allocates more learning signal to:

- minority examples
- borderline cases
- hard negatives

Although focal loss became famous in object detection, the underlying mechanism is general and applies cleanly to logistic objectives.

### How it maps to this codebase

Add a configurable loss mode:

- `loss="bce" | "weighted_bce" | "focal"`
- `focal_alpha`
- `focal_gamma`

This is still logistic regression in the sense that the model stays linear in the features and still outputs sigmoid probabilities. Only the training objective changes.

### Best use case

This is attractive when weighted BCE still overfits the many easy majority samples or when minority recall remains poor despite class weighting.

### Tradeoffs

- More hyperparameters
- Less interpretable than standard logistic likelihood
- Raw probabilities can become less calibrated than standard BCE

### Recommendation

Use this as an advanced option, not the default. It is a strong candidate when the simplest weighted approach is still not enough.

## 4. Imbalance-aware resampling with prior correction

### What to change

Train logistic regression on a better-balanced sample distribution:

- random minority over-sampling
- random majority under-sampling
- synthetic minority over-sampling such as SMOTE
- balanced mini-batches if you later move from full-batch to mini-batch optimization

### Why it helps

Resampling changes what the optimizer sees. In your current full-batch loop, majority examples dominate every step. Resampling can expose the model to a more informative class distribution and often improves minority recall dramatically.

SMOTE is particularly relevant when the minority class is too small and duplicated over-sampling would simply repeat the same few points.

### How it maps to this codebase

A practical design would be:

- keep the base logistic model
- add a preprocessing/training wrapper that creates a resampled training set
- if under-sampling or case-control sampling is used, combine it with rare-events prior correction from Approach 2

### Best fit in this repository

This is highly relevant for datasets with ratios above `20:1`, especially where the minority count is tiny.

### Tradeoffs

- Naive over-sampling can overfit
- Naive under-sampling can discard useful majority information
- SMOTE can create unrealistic synthetic points if features are categorical or poorly scaled

### Recommendation

Use stratified resampling as a training strategy around logistic regression, not as a substitute for a better loss. In practice, `weighted loss + careful resampling` is often stronger than either alone.

## 5. Tune the decision threshold and calibrate probabilities

### What to change

Stop converting probabilities to labels with a hard-coded `0.5` threshold. Instead:

- choose the threshold on a validation set
- optimize for the metric that matters: `F1`, recall at fixed precision, balanced accuracy, PR-AUC proxy, or expected cost
- calibrate probabilities before threshold selection when needed

### Why it helps

For imbalanced classification, the default `0.5` threshold is usually wrong. A model can produce useful rankings while still giving terrible class labels at `0.5`.

This is exactly what the quick baseline in this repo suggests: the model often predicts no positives at all.

Threshold tuning solves a different problem from weighted loss:

- weighted loss improves the fitted model
- threshold tuning improves the decision rule on top of that model

Calibration matters because weighting, focal loss, and resampling can distort raw probability estimates.

### How it maps to this codebase

Add:

- `predict_proba(X)`
- `predict(X, threshold=0.5)`
- validation utilities to search thresholds
- calibration support, for example Platt-style sigmoid calibration on a held-out split

Also update evaluation. Accuracy and MSE are not appropriate primary metrics here. For this project, the core metrics should be:

- precision
- recall
- F1
- balanced accuracy
- PR-AUC
- confusion matrix

### Tradeoffs

- Requires a validation protocol
- A tuned threshold is task-specific and may differ across datasets

### Recommendation

Treat this as mandatory. Even a very good imbalanced classifier can look broken if the threshold is fixed at `0.5`.

## Priority order for this project

If I were upgrading this repository incrementally, I would do it in this order:

1. Stabilize the math and API.
2. Add class-weighted BCE.
3. Add threshold tuning and proper imbalanced metrics.
4. Add resampling support.
5. Add rare-events correction.
6. Add focal loss as an advanced experimental option.

## Minimal engineering fixes that should accompany any of the five approaches

These are not counted as part of the five proposals, but they should be done immediately:

- make `sigmoid` numerically stable
- log loss during training
- expose `learning_rate`, `iterations`, and regularization as function arguments
- add L2 regularization
- stratify train/validation/test splits
- standardize numerical features before optimization
- support categorical encoding consistently

Without these fixes, it will be harder to tell whether an imbalance method failed or the optimizer simply behaved poorly.

## My recommendation

For this codebase, the best practical combination is:

1. `weighted BCE`
2. `threshold tuning on validation data`
3. `rare-events prior correction when rebalanced sampling is used`
4. `resampling for the most extreme datasets`
5. `focal loss only as an advanced option`

That combination stays close to logistic regression, preserves interpretability, and addresses the actual failure modes I observed in your datasets.

## Sources consulted

### Code and local project material

- Current implementation: [logistic_regression.py](/home/bruno-zabot/Studies/MachineLearning/assignment/logistic_regression.py)
- Example usage: [test_example.py](/home/bruno-zabot/Studies/MachineLearning/assignment/test_example.py)
- Dataset folder: `class_imbalance/`

### Articles, documentation, and references

- King, G. and Zeng, L. (2001), *Logistic Regression in Rare Events Data*  
  https://gking.harvard.edu/files/0s.pdf

- Tomz, M., King, G., and Zeng, L. (2003), *ReLogit: Rare Events Logistic Regression*  
  https://www.jstatsoft.org/article/view/v008i02

- He, H. and Garcia, E. A. (2009), *Learning from Imbalanced Data*  
  https://doi.org/10.1109/TKDE.2008.239

- Chawla, N. V., Bowyer, K. W., Hall, L. O., and Kegelmeyer, W. P. (2002), *SMOTE: Synthetic Minority Over-sampling Technique*  
  https://www.scs.cmu.edu/afs/cs.cmu.edu/project/jair/pub/volume16/chawla02a.pdf

- Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollar, P. (2017), *Focal Loss for Dense Object Detection*  
  https://arxiv.org/abs/1708.02002

- scikit-learn documentation, `compute_class_weight`  
  https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

- scikit-learn documentation, *Tuning the decision threshold for class prediction*  
  https://scikit-learn.org/stable/modules/classification_threshold.html

- scikit-learn documentation, `TunedThresholdClassifierCV`  
  https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html

- scikit-learn documentation, `CalibratedClassifierCV`  
  https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
