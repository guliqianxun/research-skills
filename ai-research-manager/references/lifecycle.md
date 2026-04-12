# AI Research Lifecycle

This skill enforces a rigorous Finite State Machine (FSM) to manage research progression. It uses a dual orthogonal state model: `lifecycle_status` (for general items) and `research_stage` (for the overarching study).

## Document State Machine (research_stage)

The `STUDY` document must linearly transition through the following stages:

- `framing`: define problem, scope, hypotheses, and success criteria
- `designing`: design experiments, baselines, budgets, and stopping rules
- `executing`: run experiments through approved tools only
- `evaluating`: compute statistical evidence and robustness checks
- `synthesizing`: assemble claim-evidence map and paper draft
- `closed`: research is concluded

Transitions are forward by default, with intentional backward loops:
- `evaluating -> designing` when evidence is missing, invalid, or too weak.
- `synthesizing -> closed` when evidence is sufficient and claims are bounded.

### Transition Validation via `previous_stage`

When advancing `research_stage`, you MUST set `previous_stage` to the old stage value before writing the new one. The validation script (`research_index.py validate`) will reject illegal transitions.

Legal transitions:

| From | To (allowed) |
|------|-------------|
| `framing` | `designing` |
| `designing` | `executing` |
| `executing` | `evaluating` |
| `evaluating` | `synthesizing`, `designing` |
| `synthesizing` | `closed` |
| `closed` | *(terminal — no transitions)* |

If `previous_stage` is empty or absent, the check is skipped (backward compatible for initial creation).

## Item State Machine (lifecycle_status)

Hypotheses and Experiments use the `lifecycle_status`:
- `proposed`: Idea drafted, pending approval or planning.
- `planned`: Ready to be executed (for experiments).
- `active`: Canonical in-progress state.
- `running`: Compatibility alias accepted by the validator for older notes; prefer `active` in new markdown.
- `paused`: Temporarily halted.
- `completed`: Successfully finished.
- `abandoned`: Intentionally stopped and retained as negative evidence.
- `failed`: Execution failed in a way worth preserving distinctly from an intentional stop.

## Claim State Machine (claim_status)

Claims use `claim_status` with automated taint-tracking:
- `drafting`: Text only, no evidence.
- `pending_evidence`: Waiting for experiments mapped in `evidence_ids` to hit `completed`.
- `validated`: All mapped evidence is `completed` and statistically sound.
- `refuted`: Mapped evidence returned negative results.
- `tainted`: If ANY underlying experiment in `evidence_ids` changes to `failed`, `abandoned`, or is modified, the claim MUST immediately downgrade to `tainted` for re-audit.

## Research Gates (required_gate)

Transitions between states are protected by gates:

### G1 Problem Gate
- Required to move from `framing` to `designing`.
- At least one `proposed` hypothesis exists.

### G2 Experiment Gate
- Required to move an experiment from `planned` to `active`.
- `required_gate: G2_approval` must be conceptually passed.
- Baseline and treatment are fully specified.

### G3 Evaluation Gate
- Required to move a claim to `validated`.
- Robustness checks are stable enough to support claims.

## Non-Negotiable Rules

1. Claims never outrun evidence.
2. Reruns must have a stated uncertainty-reduction objective.
3. Negative results stay in the record (use `abandoned` or `refuted`).
4. Markdown is the source of truth; json indexes are disposable derived views.