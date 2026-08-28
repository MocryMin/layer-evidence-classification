# EXP-004 H1 source-wise rerun protocol supplement

Freeze date: 2026-08-28 (Asia/Shanghai)

This supplement records the user's explicit operational clarifications made
after restoring the host EXP-004 document.  It does not rewrite that document.
The host document remains the scientific protocol; this file makes the exact
runner semantics and known deviations auditable before GPU execution.

## Search semantics

- S1 has fixed root `[1, 28]`; a child inserts one arbitrary layer immediately
  before the terminal layer 28.
- S2 has the canonical `[1, ..., 28]` root and uses the documented uniformly
  chosen remove/replace/swap mutation.
- S3 has the empty embedding root and appends an arbitrary layer.
- S4 has root `[1]` and appends an arbitrary layer.
- S5 has root `[28]` and inserts an arbitrary layer before terminal 28.
- The five source populations and their parent-selection temperatures are
  independent.  Paths are globally unique across sources and are attributed to
  the first source that generates them.
- Parent selection is the pure task-accuracy softmax in equation (6), without
  the earlier implementation's 0.75-softmax/0.25-uniform mixture.
- `n_s` is the number of newly generated discovery-good paths in source `s`
  during this rerun.  Fixed roots and legacy candidates do not increment it.
- The source-local temperature schedule is equation (7), with `C=1` for S2 and
  `C=0.3` for S1/S3/S4/S5.
- Fixed roots are parent-population seeds but do not count toward the 5,000 new
  candidate cap or source-wise descriptive denominators.
- Maximum path length remains 36 as an engineering safety bound.
- Validation and test remain inaccessible during discovery.

## Operational adaptation (not part of the host preregistration)

The user authorised a compute-allocation throttle after protocol drafting.  A
source with at least 100 newly generated discovery-good paths retains its
round-robin turn with probability 0.35; otherwise that turn is logged as a
throttle skip.  This adaptation is excluded from claims of strict adherence to
the original search proposal and must be disclosed in the final report.

## Legacy/new statistical boundary

- The earlier fixed-mixture tranche remains valid objective evidence that its
  discovery-good paths exist.  Its good candidates remain eligible for the
  combined H1 confirmation pool.
- Legacy candidates never initialise the new search populations, temperatures,
  source-good counters, throttle, or new source-wise denominators.
- New source-wise statistics use this rerun only.  Any combined confirmation
  pool is deduplicated by exact layer sequence and reports provenance.
- The earlier tranche's protocol deviations and this rerun's operational
  throttle are disclosed rather than retroactively described as preregistered.

## One global prefix trie

- A trie node is keyed only by the exact layer-prefix tuple.  Source identity is
  not part of the cache key because hidden state is source-independent under the
  frozen model, input split, prompt, tokenisation, dtype, and batch layout.
- Node registration records that the prefix has been traversed.  Cache payload
  state is separate and can be `none`, `partial_ssd`, `ssd`, or `hdd`.
- A new path uses the deepest complete cached prefix available anywhere in the
  global trie, including prefixes first visited by another source.
- S1/S5 admit their pre-terminal expandable prefix; S3/S4 admit their complete
  expandable path.  S2 reads global cache entries but does not automatically
  write every intermediate state of each long mutated path.
- Payloads contain all token hidden states before final RMSNorm in bfloat16.
  Last-token-only features are insufficient for continuing a decoder block.
- SSD capacity is 200 GiB and HDD capacity is 1,000 GiB.  When a tier exceeds
  capacity, the least-recently-used resident leaf payload is moved/dropped.
  Trie metadata is never removed when its payload is evicted.
- Cache shards are crash-safe and are not considered complete until both frozen
  train subsets have all expected shards.

## Stop and evidence scope

- Hard wall-clock stop: `2026-08-28T18:00:00+08:00` for the first session.
- The runner reserves 15 minutes and does not begin an expensive unit that
  cannot safely commit before the soft stop.
- Candidate safety cap: 5,000 newly generated paths.
- Discovery evidence is official-train-only (`D_fit=1750`,
  `D_discover=501`).  Official validation/test confirmation is a later stage.

## Document/operation consistency disclosure

At freeze time the host EXP-004 table still rendered S1 as `[1,...,28]`, while
the user's explicit clarification fixed it to `[1,28]`.  The host cache sentence
still said LFU, while the user's explicit clarification selected leaf-LRU for
this run.  These message-level clarifications take precedence operationally and
are recorded here rather than silently changing the host document.
