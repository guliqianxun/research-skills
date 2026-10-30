# literature-review

Reproducible literature reviews: multi-database retrieval, deduplication, relevance scoring, A/B/C
screening, open-access full-text retrieval, figure extraction, structured reading notes, and a written
review that is audited against the evidence files.

## What It Does

When you ask the AI for a literature review, survey or core-paper list (文献综述 / 文献调研), this skill provides:

- **A funnel with records at every step.** OpenAlex + Crossref + Semantic Scholar → merge and dedupe → publication-type
  classification (journal / preprint / conference abstract / thesis) → keyword scoring → AI screening into
  A/B/C tiers with reasons → core-set selection.
- **Full-text work.** Open-access PDF download with license logging, a manual-fetch checklist for paywalled
  papers, full-text extraction with page markers, figure-page rendering and a caption index.
- **Reading and structure.** Per-paper reading notes with figure/table/page for every number, a review outline for
  model-evaluation topics, ★/☆ evidence marking, and a reproducibility appendix.
- **Strict writing.** Every finding goes into a claims ledger first (source, location, evidence level, numbers).
  The draft may only use ledger claims, with wording matched to evidence strength. No citations from memory.
- **An audit step.** `check` fails the draft on:
  - uncited paragraphs
  - numbers not found in the cited papers' claims
  - unlabelled calculations
  - consensus wording backed by one study
  - author–year citations outside the evidence base
  - new evidence in the abstract or conclusions
  - ★ without full text

## Files

| File | Contents |
|------|----------|
| `SKILL.md` | Scoping, pipeline, screening, reading, writing, hard rules |
| `scripts/litreview.py` | All stages: `init search curate batches select download extract biblio check` |
| `references/pipeline.md` | Stage details, file formats, tuning |
| `references/writing-protocol.md` | Strict writing rules: claims ledger, sentence and section rules, audit |
| `references/reading-and-writing.md` | Reading notes, figures, review structure, export |
| `references/failure-modes.md` | Symptom → cause → fix, from the first real run |
| `references/example-cmip6-transient.md` | Worked example + primer on storm-track / transient-eddy diagnostics |
| `templates/topic.example.json` | Config for the worked example |
| `templates/paper_note.md`, `templates/claims.tsv`, `templates/review_outline.md` | Reading-note, claims-ledger and review templates |

## Quick Start

```bash
pip install pandas requests pymupdf openpyxl
export CONTACT_EMAIL=you@example.org          # optional: OPENALEX_API_KEY, S2_API_KEY
S=/path/to/research-skills/literature-review/scripts/litreview.py
python $S init --project ~/reviews/my-topic    # then edit ~/reviews/my-topic/topic.json
python $S search  --project ~/reviews/my-topic
python $S curate  --project ~/reviews/my-topic
python $S batches --project ~/reviews/my-topic # the AI screens batches into screen/tiers.tsv
python $S select  --project ~/reviews/my-topic
python $S download --project ~/reviews/my-topic
python $S extract --project ~/reviews/my-topic
python $S biblio  --project ~/reviews/my-topic
python $S check   --project ~/reviews/my-topic
```

## Example Prompts

- "Do a literature review on CMIP6 storm-track biases and write it in Chinese"
- "帮我调研 CMIP6 对东亚夏季风降水的模拟能力，整理核心文献清单并写综述"
- "Update my review with papers from the last year and re-run the audit"
- "Check this review draft: are all the ★ full-text claims backed by PDFs and notes?"

## Install

Claude Code: copy the folder to `~/.claude/skills/literature-review/` (all projects) or
`<project>/.claude/skills/literature-review/` (one project). Other tools: point your custom instructions at `SKILL.md`.
