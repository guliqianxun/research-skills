# Tool Contracts (IDE Agent Roles)

This skill assumes the overarching Actor is a powerful AI IDE (Antigravity, Cursor, Cline). The IDE conceptually shifts between the following roles while using its native tools (terminal, file reading, code editing) to advance the FSM.

## Shared Invariants

Every serious experiment or evaluation action should preserve these fields in the implementation code or reports:
- `code_commit`
- `config_hash`
- `data_hash`
- `seed`
- `env_fingerprint`

If any of these are missing for a decisive result, the `evaluator` role cannot support a strong claim.

## Framer Role
- Literature retrieval
- Dataset availability checks
- Power or sample-size estimation
- Falsifiability checks

The Framer focuses on the `framing` stage and updates the `STUDY` document's hypotheses.

## Planner Role
- Compute cost estimation
- Confounder detection
- Reproducibility checks

The Planner designs the `experiment` items, transitioning them to `planned`, and defines baselines before the Runner starts.

## Runner Role
The Runner is the strictest role. It uses the IDE's terminal to execute code.
Allowed action categories:
- Prepare environment
- Launch job via terminal
- Monitor job
- Attempt bounded error recovery (IDE self-correction)

Forbidden behavior:
- Publishing incomplete artifacts
- Silent retries beyond the declared retry budget (must log failures as `abandoned` or `failed`)

## Evaluator Role
- Assumption checks
- Statistical tests
- Effect-size computation

The Evaluator transitions `claim_status` based on experiment outcomes. It must emit explicit caveats when assumptions fail.

## Analyst Role
- Failure-mode classification
- Rerun prioritization
- Evidence sufficiency checks

The Analyst reviews `tainted` or `refuted` claims and decides whether to loop the `research_stage` back to `designing`.