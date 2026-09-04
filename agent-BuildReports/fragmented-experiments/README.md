# Fragmented experiments — report index

These are single-point exploratory experiments for data collection, controls,
or direction finding. They do not inherit the acceptance status of EXP-001--004
and should not be cited as confirmatory evidence unless a later frozen protocol
explicitly promotes them.

## Cross-model and cross-dataset probes

| Report | Purpose |
|---|---|
| [Qwen3-Embedding / CLINC150](Qwen3Emb0p6bExp1Ver_260812_01.md) | Side verification of intermediate-layer readout and recoverability with last-token pooling. |
| [ModernBERT / CLINC150](ModernBERTBaseExp1Ver_260812_02.md) | Side verification with a different encoder architecture and healthy CLS variance. |
| [WOS-46985 statistics](WOS46985Features_260812_03.md) | Dataset integrity, labels, length, and split statistics. |
| [DeBERTa / WOS baseline](DeBERTaV3BaseWOS46985Baseline_260812_04.md) | Tests whether the CLS compression/readout pattern recurs on a second task. |
| [ModernBERT / WOS baseline](ModernBERTBaseWOS46985Baseline_260812_05.md) | Cross-architecture WOS baseline. |
| [Qwen3-Embedding / WOS baseline](Qwen3Emb0p6bWOS46985Baseline_260814_01.md) | Last-token WOS baseline. |
| [Class-wise WOS recoverability](class-wiseRecoverabilityWOS46985_260818_01.md) | Exploratory class/domain distribution analysis. |
| [Fine-tuned DeBERTa / WOS](FT-BaselineDeBERTaV3BaseWOS46985_260818_03.md) | Full and attention-only fine-tuning control. |

## Modular/arbitrary-path probes

| Report | Purpose |
|---|---|
| [DeBERTa / CLINC modular paths](mudularized_layer_probe_260813_01.md) | Singles, pairs, greedy paths, and 4,500 random paths with path-specific ridge heads. |
| [DeBERTa / CLINC path analysis](analysis_mudularized_layer_probe_260813_01.md) | Transition utility, path-feature regression, and bigram summaries. |
| [DeBERTa / WOS modular paths](DeBERTaV3BaseWOS46985LayerProbe_260814_01.md) | Modular-path tasks on a second dataset. |
| [ModernBERT / WOS modular paths](ModernBERTBaseWOS46985LayerProbe_260814_02.md) | Modular-path tasks on a second architecture after padding correction. |
| [DeBERTa / WOS path analysis](Analysis_DeBERTaV3BaseWOS46985LayerProbe_260818_02.md) | Test replay and class-conditioned transition analysis. |

## Shared-head and translator diagnostics

| Report | Purpose | Status note |
|---|---|---|
| [Uniform shared head](uniform_head_260824_01.md) | Tests whether one canonical-size head reads many alternative paths. | Exploratory precursor to EXP-004. |
| [Low-rank canonical translator](canonical_translator_260825_01.md) | Tests low-rank residual translators before a frozen canonical head. | Bias-free and regression results stand as recorded. |
| [Translator-bias extension](translator_bias_260826_01.md) | Bias and effective-head analysis. | Original CE-with-bias evaluation withdrawn after a double-counting bug. |
| [Translator-bias corrected addendum](translator_bias_evalfix_260826_01.md) | Correct evaluation replacing only the withdrawn component. | Use this addendum for CE-with-bias accuracy. |

## Storage and provenance

- Group plans are under `user_exp_plans/fragmented_exp_*.md` and related plan
  files.
- Code is under `scripts/` and `src/`; each report records the applicable
  revision and run settings.
- Full artifacts are under
  `artifacts/fragmented-experiments/<experiment_name>/` and are gitignored.
- Initial public release policy: reports are public; raw fragmented artifacts
  remain local unless a formal claim requires a versioned secondary release.
- General protocol: [`AGENT_PROTOCOL.md`](../../AGENT_PROTOCOL.md), especially
  the fragmented-experiment reporting rules.
