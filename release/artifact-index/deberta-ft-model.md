# Fine-tuned DeBERTa model pointer

The EXP-002 author log refers to
`artifacts/EXP-20260729-002/models/deberta-v3-base-clinc150-ft/`. The historical
local model actually lives outside the artifact root under
`models/deberta-v3-base-clinc150-ft/`.

The model weights are deliberately not redistributed in the public evidence
dataset. The public evidence needed for the reported diagnostic is instead
available under `artifacts/EXP-20260729-002/03c_ft_backbone/`, including the
fine-tuning history, feature statistics, and probe results. The checkpoint can
be reproduced with the tracked fine-tuning code and recorded configuration.

This page makes the historical directory pointer resolvable without presenting
a metadata stub as if it were the omitted model itself.
