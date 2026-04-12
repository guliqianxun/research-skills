# research-skills

A collection of AI-powered skills for machine learning research — from mathematical proofs and model design to full experiment lifecycle management.

These skills work as **structured knowledge packs** for AI coding assistants. They guide the AI with decision frameworks, reference materials, templates, and validation tools so it can act as a rigorous research partner rather than a generic chatbot.

## Skills

| Skill | Description |
|-------|-------------|
| [math-proof](./math-proof/) | Construct, verify, and communicate mathematical proofs for ML theory — convergence, generalization, diffusion models, flow matching, scaling laws |
| [torch-model-design](./torch-model-design/) | PyTorch model design from zero to production — engineering principles, architecture patterns, distributed training, multimodal/temporal models, inference optimization |
| [ai-research-manager](./ai-research-manager/) | Bold hypotheses, careful verification — experiment lifecycle with FSM governance, multi-agent teams, and audit trails |

## Quick Start

### Prerequisites

- An AI agent tool that supports custom instructions (Claude Code, Cursor, Cline, opencode, aider, Windsurf, etc.)
- Python 3.10+ (for `ai-research-manager` validation script)

### Installation

**Claude Code (CLI / Desktop / Web)**

```bash
# Add skills to your project
claude mcp add-skill /path/to/research-skills/math-proof
claude mcp add-skill /path/to/research-skills/torch-model-design
claude mcp add-skill /path/to/research-skills/ai-research-manager
```

Or manually copy the skill folder into your project and reference it in your Claude Code configuration.

**Cursor / Cline / Other AI IDEs**

Copy the skill directory into your project, then reference the `SKILL.md` path in your IDE's custom instructions or system prompt configuration. See each skill's README for IDE-specific setup.

### Verify Installation

For `ai-research-manager`, run the validation script to confirm everything works:

```bash
python ai-research-manager/scripts/research_index.py build --root docs/research
```

## How Skills Work

Each skill follows a two-tier architecture:

```
SKILL.md          <- Decision hub: when/how to approach problems
  └── references/ <- Deep references: templates, code, worked examples
  └── templates/  <- Document templates (ai-research-manager)
  └── agents/     <- Role specifications (ai-research-manager)
  └── scripts/    <- Validation tools (ai-research-manager)
```

The AI reads `SKILL.md` to understand what strategy to use, then dives into reference files for implementation details. This keeps the AI focused and methodical instead of improvising.

## Skill Overview

### math-proof

Rigorous ML theory following NeurIPS/ICML/ICLR standards:

- **Proof strategy selection** — 11 strategies matched to claim patterns (induction, contradiction, coupling, amortized analysis, etc.)
- **Proof templates** — Worked examples for GD convergence, PAC-Bayes, attention Lipschitz, Adam/AMSGrad, PL condition, scaling laws
- **Generative model theory** — Diffusion SDEs, denoising score matching, flow matching with exact formulas from foundational papers
- **Advanced tools** — Rademacher complexity, optimal transport, Log-Sobolev / Talagrand, NTK, information theory, landscape analysis

[Read more](./math-proof/)

### torch-model-design

PyTorch engineering from principles to production:

- **Engineering principles** — 11 foundational rules covering computation graphs, memory, precision, training stability, parallelism, inference, multimodal conventions, arithmetic intensity, data loading, reproducibility, and loss patterns
- **Architecture patterns** — GQA, SwiGLU, RoPE, MoE, sliding window attention with decision tables
- **Distributed training** — FSDP2, tensor parallel, pipeline parallel, 2D/3D parallelism using PyTorch 2.4+ APIs (DTensor, DeviceMesh)
- **Multimodal and temporal** — Modality tokenization, cross-attention vs interleaved, alignment, streaming inference
- **Inference** — KV cache, quantization via torchao, speculative decoding, continuous batching, torch.export

[Read more](./torch-model-design/)

### ai-research-manager

A governance engine for AI-driven research that enforces auditable, reproducible workflows:

- **Finite State Machine** — Studies progress through `framing → designing → executing → evaluating → synthesizing → closed` with hard quality gates
- **Five Agent Roles** — Framer, Planner, Runner, Evaluator, Analyst — each with strict scope and handoff rules
- **Experiment Tracking** — Markdown-as-source-of-truth with YAML frontmatter, inline metadata, and a Python validation script
- **Git Integration** — Isolated branches per experiment, merge-on-success policy
- **Claim-Evidence Mapping** — Claims must link to validated experiments; taint propagation flags claims when evidence fails

[Read more](./ai-research-manager/)

## Compatibility

| Tool | math-proof | torch-model-design | ai-research-manager |
|------|:-:|:-:|:-:|
| Claude Code (CLI/Desktop/Web) | Yes | Yes | Yes |
| Cursor | Yes | Yes | Yes |
| Cline | Yes | Yes | Yes |
| Windsurf | Yes | Yes | Yes |
| Other tools with custom instructions | Yes | Yes | Yes |

These skills are tool-agnostic — any AI assistant that can read markdown instructions and execute terminal commands can use them.

## Project Structure

```
research-skills/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── math-proof/
│   ├── README.md
│   ├── SKILL.md
│   └── references/
│       ├── proof-templates.md
│       ├── generative-models.md
│       └── advanced-tools.md
├── torch-model-design/
│   ├── README.md
│   ├── SKILL.md
│   └── references/
│       ├── principles.md
│       ├── architecture.md
│       ├── profiling-optimization.md
│       ├── distributed-training.md
│       ├── multimodal-temporal.md
│       └── inference-deployment.md
└── ai-research-manager/
    ├── README.md
    ├── SKILL.md
    ├── agents/
    │   ├── framer.md
    │   ├── planner.md
    │   ├── runner.md
    │   ├── evaluator.md
    │   └── analyst.md
    ├── references/
    │   ├── lifecycle.md
    │   └── tool-contracts.md
    ├── scripts/
    │   └── research_index.py
    └── templates/
        ├── RESEARCH_ROADMAP.md
        ├── STUDY.md
        ├── EVAL_REPORT.md
        ├── COMPARISON.md
        ├── CLAIM_MAP.md
        └── PAPER_DRAFT.md
```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

[MIT](./LICENSE)
