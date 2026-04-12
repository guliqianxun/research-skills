# ai-research-manager

A hypothesis-driven research workflow skill that enforces the principle of **bold hypotheses, careful verification** (大胆假设，小心求证) across any AI agent tool.

## Philosophy

AI agents are powerful but undisciplined researchers. Without constraints, they skip hypothesis formulation, run experiments without baselines, draw conclusions from noisy data, and lose track of what was tried and why.

This skill is the discipline layer. It provides:

- A **Finite State Machine** that controls how research progresses through stages
- **Quality gates** that block advancement until criteria are met
- **Markdown-as-source-of-truth** so collaboration state survives across sessions and agents
- A **validation script** that enforces all rules automatically
- **Experiment comparison** templates for structured cross-experiment analysis

The core goal: **forward-iterating experiment content management** — keep research moving forward with clear audit trails, never getting lost in untracked changes. Failure is not waste — it is evidence.

## How It Works

### AI Agent as Engine, Skill as Constraint

Any AI agent that can read files, run terminal commands, and edit markdown can use this skill. The agent does the thinking and executing; the skill provides the rules it must follow.

Works with: **Claude Code** (CLI/Desktop/Web), **Cursor**, **Cline**, **opencode**, **aider**, **Windsurf**, and any assistant that supports custom instructions.

### Multi-Agent Team Model

Research tasks can be split across separate conversations or sessions. Agents don't need to share a conversation — they share **markdown files**.

```
Agent A (Framer)          Agent B (Planner)         Agent C (Runner)
  │ writes hypotheses       │ designs experiments      │ executes code
  └──────────┬──────────────┴──────────┬───────────────┘
             └──> docs/research/*.md <─┘
                  (validated by research_index.py)
```

Each agent loads only its role file via progressive disclosure. Shared state is solidified in markdown and validated by the Python script. Handoffs happen through state changes, not conversation context.

Single-session mode also works — one agent switches roles as the study progresses.

## Research Lifecycle

```
framing ──G1──> designing ──G2──> executing ──> evaluating ──G3──> synthesizing ──> closed
                    ^                               │
                    └───── backward loop ────────────┘
                         (insufficient evidence)
```

### Quality Gates

| Gate | Transition | Why It Exists |
|------|-----------|---------------|
| **G1** | framing → designing | Prevents vague "let's try X" from becoming unfocused experiment sprawl |
| **G2** | designing → executing | Prevents "run it and see" experiments that waste compute |
| **G3** | evaluating → synthesizing | Prevents weak or noisy results from becoming conclusions |

### Five Agent Roles

| Role | Stage | Mission |
|------|-------|---------|
| **Framer** | framing | Generate bold, falsifiable hypotheses. Scope the problem. |
| **Planner** | designing | Design experiments with baselines, budgets, and stopping rules. |
| **Runner** | executing | Execute code on isolated git branches. Record metadata. |
| **Evaluator** | evaluating | Run statistical tests and robustness checks. |
| **Analyst** | post-eval | Classify results. Decide: retry, revise, write, or stop. |

## Architecture

```
ai-research-manager/
├── SKILL.md                       # FSM rules & operating manual
├── agents/                        # Role specifications (load per-stage)
│   ├── framer.md
│   ├── planner.md
│   ├── runner.md
│   ├── evaluator.md
│   └── analyst.md
├── references/
│   ├── lifecycle.md               # FSM state diagrams & transitions
│   └── tool-contracts.md          # Script API contracts
├── scripts/
│   ├── research_index.py          # Validation & indexing (zero dependencies)
│   └── test_research_index.py     # Test suite
└── templates/
    ├── RESEARCH_ROADMAP.md        # Portfolio-level tracking
    ├── STUDY.md                   # Core study document
    ├── EVAL_REPORT.md             # Statistical evaluation
    ├── COMPARISON.md              # Cross-experiment comparison
    ├── CLAIM_MAP.md               # Claim-evidence mapping
    └── PAPER_DRAFT.md             # Paper draft
```

**Progressive disclosure**: SKILL.md is the decision hub. Agents load only their role file and relevant templates for the current stage, keeping context focused.

## Setup

### 1. Copy into your project

```bash
cp -r ai-research-manager/ /path/to/your/project/
```

### 2. Create the research directory

```bash
python scripts/research_index.py scaffold --root docs/research
```

This creates: `studies/`, `reports/`, `claim-maps/`, `comparisons/`, `drafts/`, `results/`

### 3. Verify

```bash
python scripts/research_index.py build --root docs/research
```

No external Python dependencies required.

### Tool Configuration

- **Claude Code** — automatically detected. Invoke with `/ai-research-manager` or ask research-related questions.
- **Cursor / Cline / Other** — add the path to `SKILL.md` in your custom instructions.
- **Multi-agent setup** — give each agent session only the relevant agent file (e.g., `agents/framer.md`) plus `SKILL.md`.

## Usage

### Start a new study

> "Start a new study on whether relative position bias improves long-context retrieval"

The agent enters the Framer role, creates a study document, generates hypotheses, and runs validation.

### Check gate readiness

> "Are we ready to move to the design phase?"

The agent runs the G1 gate checklist and reports what's missing.

### Compare experiments

> "Compare exp-001 and exp-002 results side by side"

The agent creates a comparison document from the template with parameter diffs, results matrices, and statistical significance.

### Snapshot experiment results

```bash
# From experiment branch, copy outputs into results/
python scripts/research_index.py snapshot --exp-id exp-001 --root docs/research
```

Auto-generates `manifest.yaml` with file checksums. Validation issues soft warnings for completed experiments missing results.

### Validation commands

```bash
python scripts/research_index.py validate --root docs/research
python scripts/research_index.py query --item_type experiment --lifecycle_status planned
python scripts/research_index.py locate --id exp-001
python scripts/research_index.py show --root docs/research
```

## Git Integration

Branch isolation ensures reproducibility:

- **Study branches** (`study/rs-001`) — created when a study starts
- **Experiment branches** (`exp/exp-001`) — branched from the study branch
- **Merge policy** — only completed + validated experiments merge back
- **Failure preservation** — failed/abandoned experiments stay on their branches for audit

## Known Limitations

- **Version comparison is structured but manual.** The COMPARISON.md template provides a format for cross-experiment analysis, but there's no automated tool to generate comparison matrices from experiment data. Users fill in the template from eval reports.
- **No built-in visualization.** The validation script produces JSON and text, not charts. Pair with W&B, MLflow, or custom dashboards for visual tracking.
- **Single backward loop.** The FSM allows only `evaluating → designing`. More complex iteration patterns require manual stage overrides.

## Requirements

- Python 3.10+ (for the validation script)
- Git (for branch isolation)
- Any AI agent tool that can read files, run commands, and edit markdown

## License

[MIT](../LICENSE)
