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
   which cases it distinguishes and what it does in each of them. Here the limit is memory: you have to
   know all its rules without looking at the picture. We call the effort for this the **Global Load**.

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

**Conditions: how long is the rule for one sample?** (adapted from *explanation size*, Laber et al.,
NeurIPS 2022)
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

For both, **lower means more interpretable**. The Global Load can be read as **the length of the rule
list** you would get if you wrote the model down as if-then rules: the model has *Cases* typical
situations (rules), and each of them is described by about *Local Load* conditions. Note that a
condition near the root of a tree appears in many rules, so it is counted several times. This is on
purpose: to describe the whole model without the picture, you need all the rules.

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

There is no objective line between "interpretable" and "not interpretable". The answer depends on who
has to understand what. So we use two thresholds, one for each question, and a grey zone instead of a
hard cut.

| Question | Measured by | Verdict |
|---|---|---|
| Can I explain **one prediction**? | Local Load | easy if ≤ 7 |
| Can I explain **the whole model**? | Global Load | ≤ 10 interpretable · 10–30 grey zone · > 30 not interpretable |

Why these numbers:

- **Local Load ≤ 7.** One prediction is one rule, and you have to keep its conditions in mind at the
  same time. Working memory holds about 7 ± 2 items (Miller, 1956), so here the classic number fits
  directly.
- **Global Load ≤ 10.** The Global Load is the length of the rule list, so "7 ± 2 items in memory" can't
  be applied literally. We chose 10 because a list of about 4–5 rules with 2 conditions each still fits
  "on a sticky note". We also calibrated it to the lecture, where a tree of depth 2 is shown as
  interpretable and a tree of depth 10 is not: in our experiments, trees of depth 2 get a Global Load
  of 4.5–7.3 and trees of depth 3 get 8–22, so 10 lies right between them.
- **Grey zone 10–30.** Here the model can still be explained piece by piece, but not as a whole in one
  go. Above 30 we call it not interpretable.

The exact numbers are our choice. In section 3 we show that the order of models does not depend on
them, only where the line is drawn.

Example: a tree of depth 5 on our noisy data has a Local Load of about 4 (every single prediction is
easy to explain) but a Global Load of 44 (nobody can describe all its rules at once). This matches the
intuition that such a tree is "easy to follow, but hard to summarise".

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

All numbers below come from `scripts/run_experiment.py`.

### Where is the threshold?

For every dataset and model family, the most complex model that still reaches each verdict
(`results/threshold.csv`):

| Dataset | Family | Interpretable (≤ 10) | Grey zone (≤ 30) | Easy prediction (Local ≤ 7) |
|---|---|---|---|---|
| iris | tree | unlimited depth | unlimited depth | unlimited depth |
| iris | forest | 1 tree | 1 tree | 1 tree |
| iris | boosting | none | 1 round | 1 round |
| wine | tree | depth 2 | unlimited depth | unlimited depth |
| wine | forest | none | 1 tree | 1 tree |
| wine | boosting | none | none | 1 round |
| breast_cancer | tree | depth 3 | unlimited depth | unlimited depth |
| breast_cancer | forest | 1 tree | 1 tree | 1 tree |
| breast_cancer | boosting | 1 round | 3 rounds | 3 rounds |
| noisy_2class | tree | depth 2 | depth 4 | depth 6 |
| noisy_2class | forest | none | 1 tree | 1 tree |
| noisy_2class | boosting | 1 round | 1 round | 3 rounds |
| noisy_4class | tree | depth 2 | depth 3 | depth 6 |
| noisy_4class | forest | none | 1 tree | 1 tree |
| noisy_4class | boosting | none | none | none |

Accuracy of the last interpretable tree compared to the best model on the dataset: iris 0.97 vs. 1.00,
wine 0.76 vs. 1.00, breast cancer 0.92 vs. 0.94, noisy_2class 0.80 vs. 0.86, noisy_4class 0.50 vs. 0.65.

What we see:

- **For single trees the model as a whole stops being interpretable at depth 2–3.** This matches the
  lecture. On clean data (iris, wine, breast cancer) even the unlimited tree is at most in the grey
  zone, because it is small anyway: the data is easy to separate.
- **Single predictions stay explainable much longer.** On the noisy data, trees up to depth 6 still
  have a Local Load ≤ 7. This is the difference between "I can follow every decision" and "I can
  describe the whole model".
- **Ensembles stop being interpretable almost immediately.** A random forest is only interpretable (or
  in the grey zone) with a single tree, gradient boosting with 1–3 rounds. A forest of only 3 trees is
  already above 30 on every dataset (e.g. 39 on iris, 194 on wine).
- **Interpretability costs accuracy, but how much depends on the data.** On breast cancer the
  interpretable tree loses only 2–3 points of accuracy. On wine it loses 24 points.
- **Multiclass gradient boosting is especially expensive**, because scikit-learn builds one tree per
  class in every round. With 4 classes, one round is already 4 trees (Global Load 210).

**Does the verdict depend on the exact threshold?** The last decision tree that counts as interpretable,
for different values of the Global Load threshold:

| Dataset | ≤ 5 | ≤ 10 | ≤ 20 | ≤ 50 |
|---|---|---|---|---|
| iris | depth 2 | unlimited | unlimited | unlimited |
| wine | depth 1 | depth 2 | unlimited | unlimited |
| breast_cancer | depth 1 | depth 3 | unlimited | unlimited |
| noisy_2class | depth 2 | depth 2 | depth 3 | depth 5 |
| noisy_4class | depth 1 | depth 2 | depth 2 | depth 4 |

The threshold only moves the line by one or two depth levels; the order of models stays the same.

Plots: `results/global_load_per_model.png` (Global Load of every model, log scale, with both lines) and
`results/accuracy_vs_global_load.png` (accuracy against Global Load, interpretable region and grey zone shaded).

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

1. **An absolute scale with a meaning.** The Global Load is the length of the rule list and the Local
   Load the length of one rule, so both can be compared to what a person can handle. A number of leaves
   or an average depth has no natural threshold, especially across model families (how many leaves of a
   forest are "too many"?).
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

- **No human study.** The thresholds are motivated by working memory and calibrated to the lecture, but
  not measured with people. This is the natural next step.
- **Some exceptions are not noise.** Real interactions (e.g. XOR) and "middle" classes also count as
  exceptions. On iris, *versicolor* lies between the other two classes, so even a depth-2 tree gets
  0.25 exceptions per sample, although a person can describe it easily.
- **Design choices.** λ = 2, the thresholds and ε = 0.001 are chosen by us. We checked λ (section 4) and
  the Global Load threshold (section 3).
- **Only structure.** The metric does not know what the features mean, and Cases saturates at the
  number of samples for huge ensembles.

---

## 6. Related work

- **Conditions** is adapted from *explanation size* (Laber et al., NeurIPS 2022). Data-weighted rule
  length is also used in *Weighted Average Depth* (Laber et al., 2023) and *CREP* (Onishi et al., 2024),
  and a user study by Piltaver et al. (2016) found leaf depth to be the main factor for understanding
  trees.
- **Local Load** is close to the operation count of Slack et al. (2019) and to *simulatability*
  (Lipton, 2018).
- **Cases:** *Born-Again Tree Ensembles* (Vidal & Schiffer, 2020) count all theoretical cells of an
  ensemble; we count only the leaf combinations the data falls into, weighted by size.
- **Exceptions:** the number of non-monotone features (*Multi-Objective Optimization of Performance and
  Interpretability of Tabular Supervised ML Models*, 2023) is a yes/no measure per feature; ours works
  per split and is weighted by data.

**What is ours:** the effective number of cases over the joint leaves of all trees, the split-level
direction check, and combining them with rule length into a local and a global load with thresholds and
a grey zone.

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
| `results/scores.csv` | accuracy, all metrics and the verdict (zone) for every (dataset, model) |
| `results/threshold.csv` | most complex model per dataset and family for each verdict |
| `results/lambda_sensitivity.csv` | rank correlations for λ = 1, 2, 3 |
| `results/metric_correlations.csv` | correlation of Global Load with the baselines |
| `results/global_load_per_model.png` | Global Load per model (log scale) |
| `results/accuracy_vs_global_load.png` | accuracy against Global Load |
| `results/tree_components.png` | the three parts for single trees of growing depth |

All datasets ship with scikit-learn or are generated, so no download is needed.

### Using the metric on your own model

```python
from iml_interpretability import Dataset, GlobalLoad, LocalLoad, interpretability_zone

data = Dataset(name="my_data", X=X_train, y=y_train)   # X_train: pandas DataFrame, numeric
model = ...fit(X_train, y_train)                       # DecisionTree / RandomForest / GradientBoosting classifier
global_load = GlobalLoad().score(model, data)
print(global_load, interpretability_zone(global_load), LocalLoad().score(model, data))
```

---

## 8. Repository structure

```
iml_interpretability/
├── data.py        # datasets (iris, wine, breast cancer, noisy synthetic)
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

> **TODO (group): check and adapt this statement before submitting.**

We used Claude (Anthropic, Claude Opus 5.5) during this exercise. We used it to understand the task,
to brainstorm metric ideas and check whether similar metrics already exist in the literature, to
implement the metric, the experiments and the tests, and to draft this README. The idea to combine the
effective number of cases with the direction check, and the distinction between tracing a prediction
and understanding the model, came up in our discussions with it. We reviewed the code and the results,
ran the experiments ourselves, and checked that the numbers in this README match our results files.
[Describe here how you made the slides.] We take full responsibility for the content of this repository.
