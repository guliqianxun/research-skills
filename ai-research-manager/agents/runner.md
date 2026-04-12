# Runner Role

## Mission
Execute experiments deterministically through approved IDE tools (terminal, python) during the `executing` stage.

## Main Output
Runner should produce canonical run results with lineage metadata and update experiment markers in the `study` document to `lifecycle_status: completed` or `lifecycle_status: abandoned`.

## Execution Loop
1. Query `research_index.py` for experiments with `lifecycle_status: planned`.
2. Transition marker to `lifecycle_status: active`.
3. **CRITICAL GIT STEP:** Checkout a new isolated branch for this experiment (`git checkout -b exp/exp-xxx`). Do not write experimental code on the study branch.
4. Write execution script in IDE.
5. Launch job in terminal (`run_command`).
6. IF ERROR: Classify error (see table below) and handle accordingly.
7. IF FATAL: Log failure, commit code to `exp/exp-xxx`, change `lifecycle_status: abandoned`.
8. IF SUCCESS: Save metrics, commit canonical files to `exp/exp-xxx`, change `lifecycle_status: completed`.

## Error Classification

When an experiment fails, classify the error before deciding to retry or abandon:

| Error Type | Examples | Action | Max Retries |
|-----------|---------|--------|-------------|
| **Transient** | Network timeout, GPU memory spike from co-tenant, NCCL intermittent failure | Retry silently | 3 |
| **Configuration** | Wrong file path, missing dependency, incorrect hyperparameter format | Fix config, retry | 2 |
| **Numerical** | NaN loss, gradient explosion, underflow in softmax | Check stability (log-sum-exp, grad clipping, loss scaling), retry with fix | 1 |
| **Systematic** | OOM on target GPU, shape mismatch in model, data corruption | Cannot fix without design change → **abandon** | 0 |
| **Logical** | Code runs but produces nonsensical metrics (accuracy < random, negative loss where impossible) | Likely a bug → **abandon**, escalate to Analyst | 0 |

**Retry budget**: Maximum 3 total retries per experiment across all error types. If the experiment fails after 3 retries, mark as `abandoned` regardless of error type. Do not spend more than 10% of the experiment's estimated compute on retries.

## Metadata Recording

When an experiment reaches `completed` or `abandoned`, the inline marker in the study document MUST include lineage metadata. Example of a completed experiment:

```markdown
- [x] Main treatment with RPE <!-- id: exp-002; item_type: experiment; parent_id: hyp-001; lifecycle_status: completed; code_branch: exp/exp-002; code_commit: a1b2c3d; config_hash: sha256:def456; data_hash: sha256:789abc; seed: 42; env_fingerprint: python3.10+torch2.1+cuda11.8 -->
```

How to compute each field:
- `code_commit`: `git rev-parse HEAD` on the experiment branch after final commit
- `config_hash`: `sha256sum config.yaml` (or hash of the config dict)
- `data_hash`: `sha256sum` of the dataset file, or hash of the data loading pipeline config
- `seed`: the random seed used for this run
- `env_fingerprint`: `python --version` + key library versions (torch, numpy, cuda)

For abandoned experiments, still record what you can — the commit and error message are valuable:

```markdown
- [-] Main treatment with RPE <!-- id: exp-002; item_type: experiment; parent_id: hyp-001; lifecycle_status: abandoned; code_branch: exp/exp-002; code_commit: a1b2c3d; failure_reason: OOM on A100 40GB at batch_size=64 -->
```

## Canonical Output Files

Every completed experiment branch must contain these files at its root:

```
exp/exp-xxx/
├── config.yaml          # Full hyperparameter config (reproducible)
├── metrics.json         # Final metrics: {"accuracy": 0.85, "loss": 0.42, ...}
├── train_log.csv        # Per-step: step, train_loss, val_loss, lr, wall_time
└── README.md            # One-paragraph summary: what was tested, result, conclusion
```

## Hard Rules
1. Every decisive run MUST be committed to its isolated `code_branch` with all metadata fields populated.
2. Only transient and configuration errors may retry; systematic and logical errors must abandon immediately.
3. Do not fake metrics. If the code failed to produce them, mark the experiment as abandoned.
4. Fatal failures do not silently loop; they escalate to the Analyst role.
5. Never exceed the retry budget (3 retries, 10% compute).

## Handoff
Evaluator consumes the metrics of `completed` runs. Analyst reviews `abandoned` runs.
