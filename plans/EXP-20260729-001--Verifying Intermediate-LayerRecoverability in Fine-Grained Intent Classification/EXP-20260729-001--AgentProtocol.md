# EXP-20260729-001 — Agent Protocol

## 1. Scope

Implement the **frozen-backbone pilot** for intermediate-layer recoverability on `CLINC150` with `DeBERTa-v3-base`.

This protocol covers:

1. dataset preparation;
2. one-pass hidden-state caching;
3. independent linear-head training for Transformer layers 1–12;
4. validation-based checkpoint selection;
5. one-time test evaluation;
6. recoverability, oracle, class-wise, and confidence-interval analysis;
7. reproducible artifact export.

Out of scope for this run: QK-logits fusion, task-adapted backbone probing, `Qwen3-Embedding-0.6B`, and ModernBERT.

---

## 2. Fixed Configuration

```yaml
experiment_id: EXP-20260729-001
model_name: microsoft/deberta-v3-base
dataset: CLINC150
dataset_split: official
pooling: cls
max_length: 512
truncation: left
lowercase: false
static_padding: false
dynamic_padding: longest_in_batch
sentencepiece_version: 0.2.2

head_type: linear_with_bias
loss: cross_entropy
backbone_frozen: true
learning_rate: 1.0e-5
optimizer: AdamW
weight_decay: 0.01
epochs: 100
checkpoint_frequency: every_epoch
checkpoint_selection:
  primary: validation_accuracy
  tie_breaker: validation_nll
batch_size_cached_head_training: 256
gradient_clip_norm: 1.0
scheduler: none

seeds: [17, 29, 43, 59, 71, 89, 101, 127, 149, 173]
```

Use the **same seed list for every layer**, including layer 12. Within seed \(s\), all layers must use the same head-initialization seed and cached-data shuffle seed, so layer-wise differences are paired.

---

## 3. Input Contract

Use the same minimal prompt for all splits:

```text
Classify the intent: {utterance}
```

Tokenization requirements:

- do not lowercase;
- use left truncation;
- use `max_length=512`;
- do not persist fixed-length padding;
- use dynamic batch padding with a correct `attention_mask`;
- save `label2id.json` and `id2label.json`;
- preserve the official train/validation/test split.

---

## 4. Hidden-State Contract

Run the frozen backbone once for each split with:

```python
output_hidden_states=True
```

Hugging Face indexing:

```text
outputs.hidden_states[0]  = embedding output
outputs.hidden_states[1]  = Transformer block 1 output
...
outputs.hidden_states[12] = Transformer block 12 / final-layer output
```

For every Transformer layer \(l\in\{1,\dots,12\}\), extract:

\[
h_l(x)=H_l(x)[:,0,:]
\]

where index 0 is the `CLS` token.

Cache hidden states and labels once. Do not rerun the backbone while training heads.

Recommended cache format: `safetensors` or memory-mapped NumPy, stored in `float16`; convert batches to `float32` during head training.

---

## 5. Head-Training Contract

For every seed \(s\) and layer \(l\in\{1,\dots,12\}\):

1. initialize an independent linear head
   \[
   W_E^l\in\mathbb R^{768\times150}
   \]
   with bias;
2. use Xavier-uniform weight initialization and zero bias;
3. train only this head on cached train features;
4. evaluate validation accuracy and NLL after every epoch;
5. save an epoch-end checkpoint;
6. select the checkpoint with best validation accuracy, breaking ties by lower validation NLL;
7. load the selected checkpoint;
8. evaluate the test split exactly once;
9. save complete logits, probabilities, predictions, and per-sample metrics.

Do not select layers, epochs, thresholds, or hyperparameters using test results.

---

## 6. Required Metrics

For every seed and layer, compute:

- accuracy;
- macro-F1;
- NLL;
- probability margin \(p_{(1)}-p_{(2)}\);
- logit margin \(z_{(1)}-z_{(2)}\);
- gold margin \(z_y-\max_{c\ne y}z_c\);
- predictive entropy;
- ECE with 10 equal-width confidence bins;
- confusion matrix;
- \(R_l\), \(H_l\), \(R_{l,c}\), and \(H_{l,c}\);
- oracle accuracy and oracle recoverability;
- normalized class-wise recoverability divergence \(D_{JS}^{class}\).

Entropy:

\[
\operatorname{Entropy}_l(x)
=
-\sum_{c=1}^{C}p_{l,c}(x)\log p_{l,c}(x)
\]

Recoverability:

\[
R_l=P(\hat y_l=y\mid \hat y_L\ne y)
\]

Class-wise recoverability:

\[
R_{l,c}
=
P(\hat y_l=y\mid y=c,\hat y_L\ne y)
\]

Oracle quantities:

\[
\operatorname{Acc}_{oracle}
=
P\left(\hat y_L=y\ \lor\ \exists l<L:\hat y_l=y\right)
\]

\[
R_{oracle}
=
P\left(\exists l<L:\hat y_l=y\mid\hat y_L\ne y\right)
\]

Verify automatically:

\[
\operatorname{Acc}_{oracle}
=
\operatorname{Acc}_L+
(1-\operatorname{Acc}_L)R_{oracle}
\]

For class-wise ratios, always save numerator and denominator. Use `NA`, not zero, when the denominator is zero.

---

## 7. Hypothesis Judgement

Use validation results to select confirmatory layers; all-layer test results remain exploratory.

### H1 — Intermediate-layer non-inferiority

For seed \(s\) and intermediate layer \(l<L\), define:

\[
d_{1,s}(l)=\operatorname{Acc}_{L,s}-\operatorname{Acc}_{l,s}
\]

with non-inferiority tolerance:

\[
\epsilon_1=0.02
\]

Select the candidate intermediate layer using mean validation accuracy. H1 is supported on test only when the upper bound of the paired 95% seed-bootstrap confidence interval satisfies:

\[
CI^{upper}_{95\%}\bigl(d_1(l)\bigr)<\epsilon_1
\]

Do **not** test whether \(\epsilon_1\) lies inside the confidence interval.

### H1' — Intermediate-layer superiority

Define:

\[
d_{2,s}(l)=\operatorname{Acc}_{l,s}-\operatorname{Acc}_{L,s}
\]

Select the candidate layer using validation results. H1' is supported on test only when:

\[
CI^{lower}_{95\%}\bigl(d_2(l)\bigr)>0
\]

A positive mean whose confidence interval crosses zero is reported as a trend, not confirmation.

### H2 — Corrective recoverability

For seed \(s\), define the oracle accuracy gain:

\[
g_s=\operatorname{Acc}_{oracle,s}-\operatorname{Acc}_{L,s}
\]

H2 has statistical support when:

\[
CI^{lower}_{95\%}(g)>0
\]

Use the following as a **strong continuation criterion**, rather than as the statistical definition of recoverability:

\[
\overline g> \epsilon_2,\qquad \epsilon_2=0.05
\]

Since the CLINC150 test split contains 4,500 samples, a five-percentage-point oracle gain already corresponds to at least 225 potentially recoverable samples. Do not add a separate \(T=100\) threshold.

Report the following practical quantities:

- mean oracle gain \(\overline g\);
- \(R_{oracle}\);
- recoverable sample count;
- normalized class-wise recoverability divergence \(D_{JS}^{class}\).

#### Class-wise recoverability divergence

Raw recoverable counts are confounded by how frequently the final layer fails on each class. Therefore, compare the class distribution of recoverable errors against the class distribution of all final-layer errors.

For seed \(s\) and class \(c\), define:

\[
n^{err}_{s,c}
=
\left|
\left\{
x:y=c,\ \hat y_{L,s}\neq y
\right\}
\right|
\]

\[
n^{rec}_{s,c}
=
\left|
\left\{
x:y=c,\ \hat y_{L,s}\neq y,\ 
\exists l<L:\hat y_{l,s}=y
\right\}
\right|
\]

Construct the distributions:

\[
e_{s,c}
=
\frac{n^{err}_{s,c}}
{\sum_j n^{err}_{s,j}}
,\qquad
r_{s,c}
=
\frac{n^{rec}_{s,c}}
{\sum_j n^{rec}_{s,j}}
\]

and:

\[
m_{s,c}=\frac{e_{s,c}+r_{s,c}}{2}
\]

The normalized Jensen–Shannon divergence is:

\[
D_{JS,s}^{class}
=
\frac{
\frac12 D_{KL}(e_s\Vert m_s)
+
\frac12 D_{KL}(r_s\Vert m_s)
}{
\log 2
}
\in[0,1]
\]

Use the convention \(0\log 0=0\). If a seed has no final-layer errors or no recoverable errors, record the divergence as `NA`.

Interpretation:

- \(D_{JS}^{class}=0\): recoverable errors follow the same class distribution as final-layer errors;
- larger \(D_{JS}^{class}\): recoverability is more class-dependent and disproportionately concentrated in some classes.

Report the mean, standard deviation, and 95% seed-bootstrap confidence interval of \(D_{JS}^{class}\). Do not use a fixed divergence threshold to accept or reject H2 in this pilot.

### Confidence-interval procedure

Use the same 10 seeds for every layer. Primary intervals:

- paired seed bootstrap for \(d_1(l)\), \(d_2(l)\), and \(g\);
- ordinary seed bootstrap for \(D_{JS}^{class}\);
- 10,000 bootstrap resamples;
- percentile 95% confidence intervals.

Optional secondary analysis: paired sample bootstrap within each seed.

---

## 8. Artifact Schema

```text
artifacts/
├── README.md
├── run_config.yaml
├── seeds.json
├── label2id.json
├── id2label.json
├── cache/
│   ├── train_hidden.safetensors
│   ├── validation_hidden.safetensors
│   ├── test_hidden.safetensors
│   └── cache_manifest.json
├── checkpoints/
│   └── seed_<seed>/
│       └── layer_<01-12>/
│           ├── epoch_<001-100>.pt
│           └── best_checkpoint.json
├── predictions/
│   └── seed_<seed>_test.parquet
├── metrics/
│   ├── layer_metrics.csv
│   ├── classwise_recovery.csv
│   ├── oracle_summary.csv
│   ├── class_recovery_divergence.csv
│   ├── confidence_intervals.json
│   └── hypothesis_judgement.json
├── confusion_matrices/
└── plots/
```

`predictions/seed_<seed>_test.parquet` must contain at least:

```text
sample_id
text
gold_label
layer
logits
probabilities
prediction
nll
probability_margin
logit_margin
gold_margin
entropy
```

The root `README.md` must explain:

- every file and directory;
- file formats and schemas;
- layer indexing;
- seed policy;
- checkpoint-selection rule;
- metric definitions, including the exact class-wise Jensen–Shannon divergence construction;
- how to reproduce summary tables from raw predictions.

---

## 9. Reproducibility and Logging

- record the Git commit;
- record package versions, CUDA/PyTorch/Transformers versions, and `sentencepiece==0.2.2`;
- save the complete resolved configuration;
- save the generated cache manifest with shapes, dtypes, split sizes, and checksums;
- create one MLflow parent run and nested runs for each seed/layer;
- log validation metrics per epoch and final test metrics once;
- fail loudly on NaN/Inf logits, label-map mismatch, sample-order mismatch, or oracle-identity failure.

---

## 10. Completion Criteria

The run is complete only when:

1. all three splits are cached;
2. all 12 layers × 10 seeds have a selected checkpoint;
3. test predictions exist for every layer and seed;
4. all required metric and oracle files are generated;
5. confidence intervals and hypothesis decisions are exported;
6. `README.md` fully documents the artifact root;
7. the experiment log is updated with Git commit, MLflow run ID, artifact path, data, model, and seed list.
