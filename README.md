# iML 26/27 – Exercise 1: Interpretability Load

We propose a metric that measures how interpretable a tree-based model is: the **Interpretability Load**.
It counts how much a person has to deal with to understand the model. It works for single decision
trees, random forests and gradient-boosted trees, for binary and multiclass problems.

This README explains the idea step by step, shows our results and says how to reproduce them.

---

## 1. The idea

### What do we mean by "interpretable"?

We call a model interpretable if a person can understand how it makes its decisions. To turn this into
a number, we count how many things a person has to deal with. There are two different situations:

1. **Tracing one prediction (with pen and paper).** You take one sample and follow the model step by
   step until you get the answer. Here the limit is your time and patience, not your memory. With
   enough paper you can do this for *any* model, even a random forest with 100 trees. It just takes long.
   We call the effort for this the **Local Load**.

2. **Understanding the whole model (in your head).** You want to know how the model behaves in general:
   which cases it distinguishes and what it does in each of them. Here the limit is working memory.
   People can keep only about 7 ± 2 things in mind at once (Miller, 1956). We call the effort for this
   the **Global Load**.

The difference matters. Following 100 trees on paper gives you the answer, but you still don't
understand *why* the model decides that way. So a model can be traceable without being interpretable.
We report both numbers and use the Global Load as our main metric.

### The three ingredients

**Cases: how many typical situations does the model have?**
Every sample ends up in exactly one leaf of every tree. We call the combination of these leaves the
sample's *case*. For a single tree a case is just a leaf. For a forest it is a combination of leaves
across all trees, so the number of cases grows very quickly with the number of trees.

We do not simply count the cases. A case that only one sample reaches is not as important as a case
that half of the data falls into. So we use the *effective* number of cases:

    Cases = exp( - sum_c p_c * log(p_c) )      p_c = fraction of samples in case c

This is the exponential of the entropy. If the samples are spread equally over 4 cases, Cases = 4.
If 95 % of the samples are in 2 cases and a few samples are in 10 other cases, Cases is only a bit
above 2.

**Conditions: how long is the rule for one sample?**
For one sample we count how many *different* features are checked on its way from the root to the leaf.
A feature that is checked twice (e.g. `age < 50`, then `age < 30`) counts only once, because a person
reads this as one condition (`age < 30`). For ensembles we add this up over all trees. Then we take
the average over all samples.

**Exceptions: does the model contradict itself?**
Usually a feature has one general direction, for example "the older, the higher the risk". A person
can remember this as one simple rule. But a tree can also contain a split that goes the other way:
"in this branch, older means *lower* risk". This cannot be derived from the general rule, so it has to
be remembered separately. We call such a split an **exception**.

How we find exceptions:

1. For every split we look at how the prediction changes when we go to the branch with the larger
   feature value (`x > t`). For classification trees the prediction is the fraction of each class in
   the node, for gradient boosting it is the boosting score of the tree's class.
2. For every pair (feature, class) we decide the *usual direction*: we add up these changes over all
   splits on that feature in all trees, weighted by the number of samples in the split. The sign of the
   sum is the usual direction.
3. A split gets a contradiction score between 0 and 1: the part of its effect that goes against the
   usual direction, divided by its whole effect (summed over classes). In the binary case the score is
   always exactly 0 (fine) or 1 (exception). With more classes a split can be partly an exception, for
   example if it agrees with the usual direction for two classes but not for the third one.
   Changes smaller than 0.001 are ignored.
4. For every sample we add up the contradiction scores of the splits on its paths. Then we take the
   average over all samples.

### Putting it together

    Local Load  = Conditions + 2 × Exceptions     → effort to trace one prediction
    Global Load = Cases × Local Load              → effort to understand the whole model

For both, **lower means more interpretable**. The Global Load can be read as "the number of conditions
you have to keep in mind": the model has *Cases* typical situations, and each of them is described by
about *Local Load* conditions.

Exceptions count double, because they are harder to remember than normal conditions. The factor 2 is
our choice. In section 4 we show that the ranking of models stays almost the same for 1 or 3.

### A small example

A decision tree of depth 2 on the breast cancer data:

- the samples are spread over the leaves very unequally, so **Cases = 2.8**,
- every sample is checked on 2 different features, so **Conditions = 2.0**,
- no split goes against its feature's usual direction, so **Exceptions = 0**,
- **Local Load = 2.0** and **Global Load = 2.8 × 2.0 = 5.6**.

A random forest with 100 trees of depth 3 on the same data:

- Cases = 185, Conditions = 277, Exceptions = 5.0,
- Local Load = 287 (traceable on paper, but it takes a while),
- Global Load = 53,206 (nobody can keep this in mind).

### Threshold of interpretability

Because the Global Load is measured in "conditions to keep in mind", we can relate it to the
7 ± 2 limit of working memory. We use the upper end of this range plus a bit of slack and call a model
**interpretable if its Global Load is at most 10**. This threshold is a hypothesis based on psychology,
not something we measured with people (see section 5).

---

## 2. Setup

**Models.** All from one model class (trees and tree ensembles), with complexity growing step by step:

| Family | Variants |
|---|---|
| Decision tree | max depth 1, 2, 3, 4, 5, 6, 8, 10 and unlimited |
| Random forest | 1, 3, 10, 30, 100 trees of depth 3 |
| Gradient boosting | 1, 3, 10, 30, 100 rounds of depth-2 trees |

**Datasets.**

| Dataset | Samples | Features | Classes | Why |
|---|---|---|---|---|
| iris | 150 | 4 | 3 | very easy, classic |
| wine | 178 | 13 | 3 | small multiclass |
| breast_cancer | 569 | 30 | 2 | real binary data, many features |
| noisy_2class | 2000 | 10 | 2 | synthetic, 10 % of labels flipped |
| noisy_4class | 2000 | 10 | 4 | synthetic multiclass, 10 % of labels flipped |
| titanic | 1309 | 7 (before one-hot encoding) | 2 | from OpenML; skipped if it can't be downloaded |

The noisy datasets show what happens when a model starts fitting noise.

**Protocol.** 75 % / 25 % stratified train/test split with seed 0. Accuracy is measured on the test
set. All interpretability metrics are computed on the training data, because that is the data the
model was built from.

**Baselines.** We compare our metric with three simpler measures:

- `n_leaves`: total number of leaves over all trees,
- `avg_depth`: average number of tests on a sample's paths (= weighted average depth, as in
  Laber et al. 2023 and similar to the operation count of Slack et al. 2019),
- `simulation_effort`: tests + feature switches on a sample's path. This idea came up in our group, so
  we included it as a baseline.

---

## 3. Results

All numbers below come from `scripts/run_experiment.py` (results for the five datasets that ship with
scikit-learn; Titanic is added automatically if OpenML is reachable).

### Where is the threshold?

The most complex model of each family that is still interpretable (Global Load ≤ 10):

| Dataset | Decision tree | Random forest | Gradient boosting | Accuracy of that tree | Best accuracy of any model |
|---|---|---|---|---|---|
| iris | unlimited depth (9.1) | 1 tree (7.6) | none | 0.97 | 1.00 |
| wine | depth 2 (6.2) | none | none | 0.76 | 1.00 |
| breast_cancer | depth 3 (9.5) | 1 tree (8.9) | 1 round (5.6) | 0.92 | 0.94 |
| noisy_2class | depth 2 (4.7) | none | 1 round (4.7) | 0.80 | 0.86 |
| noisy_4class | depth 2 (7.3) | none | none | 0.50 | 0.65 |

What we see:

- **For single trees the threshold is at depth 2–3.** This matches the lecture, where a tree of depth 2
  is shown as interpretable and a tree of depth 10 as not. Iris is the exception: its full tree is
  small anyway (Global Load 9.1), so even the unlimited tree is interpretable.
- **Ensembles stop being interpretable almost immediately.** A random forest is only interpretable with
  a single tree, gradient boosting with at most one round. Already 3 trees push the Global Load far
  above the threshold (e.g. 39 on iris, 194 on wine).
- **Interpretability costs accuracy, but how much depends on the data.** On breast cancer the
  interpretable tree loses only 2–3 points of accuracy. On wine it loses 24 points.
- **Multiclass gradient boosting is especially expensive**, because scikit-learn builds one tree per
  class in every round. With 4 classes, one round is already 4 trees (Global Load 210).

Plots: `results/global_load_per_model.png` (Global Load of every model, log scale) and
`results/accuracy_vs_global_load.png` (accuracy against Global Load, interpretable region shaded).

### What the parts tell us

`results/tree_components.png` shows the three parts for single trees of growing depth.

- On the clean datasets (wine, breast cancer) the parts stop growing after depth 4. The deeper trees
  only add leaves with very few samples, so the effective number of cases barely changes.
- On the noisy datasets all parts keep growing, and **exceptions appear exactly when the tree starts to
  fit noise**. On noisy_2class the trees up to depth 4 have no exceptions. From depth 5 on they do
  (0.22 at depth 5, 1.79 at depth 10), while test accuracy stays between 0.82 and 0.85 and drops to 0.79
  for the unlimited tree. The model's logic gets messier without getting better.

### How does our metric compare to simpler measures?

`results/metric_correlations.csv` shows the Spearman rank correlation between Global Load and the
baselines over all models of a dataset:

| Dataset | vs. n_leaves | vs. avg_depth | vs. simulation_effort |
|---|---|---|---|
| iris | 1.00 | 1.00 | 1.00 |
| wine | 0.99 | 0.99 | 0.99 |
| breast_cancer | 0.94 | 0.97 | 1.00 |
| noisy_2class | 0.95 | 0.99 | 0.99 |
| noisy_4class | 0.92 | 1.00 | 1.00 |

We want to be honest about this: **when models differ in size by several orders of magnitude, every
reasonable measure ranks them in almost the same order.** A forest with 100 trees is worse than a stump
no matter how you count. So the ranking alone does not show that our metric is better.

What our metric adds:

1. **An absolute scale with a meaning.** "10 conditions to keep in mind" can be compared to working
   memory. A number of leaves or an average depth has no natural threshold, especially across model
   families (how many leaves of a forest are "too many"?).
2. **It only counts what the data actually uses.** On breast cancer, the tree of depth 5 has 18 leaves
   instead of 13 for depth 4, so `n_leaves` says it is worse. But the extra leaves contain almost no
   samples, and our Global Load stays the same (11.8 vs. 12.2).
3. **It sees self-contradicting logic.** On noisy_2class, a decision tree of depth 6 and gradient
   boosting with 3 rounds have almost the same average depth (5.98 vs. 6.0). But the tree has more
   cases (14.3 vs. 7.9) and contradicts itself (0.74 exceptions per sample vs. 0), so its Global Load is
   more than twice as high (88 vs. 40).

We also checked the `simulation_effort` idea from our group: it correlates with plain `avg_depth` at
0.97–1.00, because in a tree two splits in a row on the same feature are rare, so the number of
feature switches is almost always "tests − 1".

---

## 4. Robustness: the weight of exceptions

The factor 2 for exceptions is our choice. `results/lambda_sensitivity.csv` shows the Spearman
correlation between the model rankings for λ = 1, 2, 3: it is **at least 0.996 on all datasets**.
The set of interpretable models (Global Load ≤ 10) is also the same for λ = 1 and 2 on all datasets.
Only on iris it shrinks for λ = 3 (from 10 to 4 models), because the iris trees have exceptions already
at depth 2 (see section 5).

---

## 5. Limitations

- **No experiment with people.** The task does not require a human study for a metric that is not
  LLM-based, and we did not have time for one. The threshold of 10 is motivated by the "7 ± 2" limit of
  working memory and matches the intuition from the lecture, but it is not measured. The parts of our
  metric are based on findings from user studies (Piltaver et al. 2016 found that the depth of leaves
  matters most for how well people understand trees). A small study where people trace and describe
  trees of different Global Load would be the natural next step. A group member already started a
  prototype app for this.
- **"Middle" classes count as exceptions.** On iris, the class *versicolor* lies between the other two:
  with growing petal size its probability first goes up and then down. Our metric counts this as an
  exception (0.25 per sample already at depth 2), even though a person can describe it easily
  ("small → setosa, medium → versicolor, large → virginica"). With ordered classes the metric is
  therefore a bit too strict.
- **Real interactions also count as exceptions.** If the effect of a feature really depends on another
  feature (e.g. XOR), the tree has to contradict itself. We think this is fine, because such models are
  really harder to explain, but it means that exceptions do not always mean noise.
- **Cases saturates for huge ensembles.** For a forest with 100 trees almost every sample has its own
  case, so Cases can't grow beyond the number of samples. These models are far above the threshold
  anyway.
- **Only structure, no meaning.** The metric does not know what the features mean. `age < 30` and a
  condition on an obscure engineered feature count the same.
- **Design choices.** The weight λ = 2, the threshold of 10 and ε = 0.001 are chosen by us. We checked
  λ (section 4), but not the other two.

---

## 6. Related work

- **Depth and rule length.** Piltaver et al. (2016, *What makes classification trees comprehensible?*)
  found in a user study that the depth of leaves is the most important factor. Data-weighted depth is
  used as *Weighted Average Depth* (Laber et al., 2023), the number of distinct features on a path as
  *explanation size* (Laber et al., NeurIPS 2022), and coverage-weighted rule length as *CREP*
  (Onishi et al., 2024). Our *Conditions* part builds on these.
- **Operation count.** Slack et al. (2019, *Assessing the Local Interpretability of ML Models*) count
  the operations needed to compute a prediction by hand and show in a user study that people make more
  mistakes when this number grows. This is close to our Local Load. Lipton (2018) calls the ability to
  step through a model in reasonable time *simulatability*.
- **Ensembles as one tree.** *Born-Again Tree Ensembles* (Vidal & Schiffer, 2020) count the cells of an
  ensemble over the whole feature space to build an equivalent single tree. Our *Cases* instead counts
  the leaf combinations the data actually falls into, weighted by how many samples are in each.
- **Monotonicity.** *Multi-Objective Optimization of Performance and Interpretability of Tabular
  Supervised Machine Learning Models* (2023) uses the number of non-monotone features as a yes/no
  measure per feature. Our *Exceptions* works per split, can be partly an exception, and is weighted by
  the data.
- **Functional complexity.** Molnar et al. (2019) measure number of features, interaction strength and
  main effect complexity with a functional decomposition.

**What is ours:** the effective number of cases over the joint leaves of all trees, the split-level
direction check, and combining both with rule length into a local and a global load with a threshold
in "conditions to keep in mind". To our knowledge this combination has not been proposed before.

---

## 7. How to reproduce

We recommend [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/run_experiment.py
```

With plain pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/run_experiment.py
```

The experiment takes less than a minute. It writes:

| File | Content |
|---|---|
| `results/scores.csv` | accuracy and all metrics for every (dataset, model) |
| `results/threshold.csv` | most complex interpretable model per dataset and family |
| `results/lambda_sensitivity.csv` | rank correlations for λ = 1, 2, 3 |
| `results/metric_correlations.csv` | correlation of Global Load with the baselines |
| `results/global_load_per_model.png` | Global Load per model (log scale) |
| `results/accuracy_vs_global_load.png` | accuracy against Global Load |
| `results/tree_components.png` | the three parts for single trees of growing depth |

Titanic is downloaded from OpenML on the first run and cached in `data/`. If the download fails, it is
skipped with a message and the other datasets are still evaluated.

### Using the metric on your own model

```python
from iml_interpretability import Dataset, GlobalLoad, LocalLoad

data = Dataset(name="my_data", X=X_train, y=y_train)   # X_train: pandas DataFrame, numeric
model = ...fit(X_train, y_train)                       # DecisionTree / RandomForest / GradientBoosting classifier
print(GlobalLoad().score(model, data), LocalLoad().score(model, data))
```

---

## 8. Repository structure

```
iml_interpretability/
├── data.py        # datasets (iris, wine, breast cancer, noisy synthetic, Titanic)
├── models.py      # model grid: trees of growing depth, forests and boosting of growing size
├── trees.py       # helpers that look inside fitted tree models (works for all three families)
├── metric.py      # Cases, Conditions, Exceptions, LocalLoad, GlobalLoad + baselines
└── evaluate.py    # fits every model on every dataset and computes all metrics
scripts/
└── run_experiment.py   # runs everything, writes tables and plots to results/
tests/
└── test_smoke.py       # offline tests (known values, monotonic behaviour, all model types)
presentation/
└── presentation.pdf    # our slides
```

---

## AI statement

> **TODO (group): check and adapt this statement.**

We used Claude (Anthropic, Claude Opus 5.5) during this exercise. We used it to understand the task,
to brainstorm metric ideas and check whether similar metrics already exist in the literature, to
generate code, and to draft this README. The idea to combine the
effective number of cases with the direction check, and the distinction between tracing a prediction
and understanding the model, came up in our discussions with it. We reviewed the code and the results,
ran the experiments ourselves, and checked that the numbers in this README match our results files.
We take full responsibility for the content of this repository.
