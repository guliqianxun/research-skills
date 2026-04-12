# Contributing to research-skills

Thanks for your interest in contributing! This project is a collection of AI skills for ML research, and we welcome improvements from the community.

## Ways to Contribute

- **Bug reports** — If a skill gives bad advice, has outdated references, or the validation script fails unexpectedly, open an issue.
- **Skill improvements** — Better decision trees, new reference material, additional templates, or improved validation logic.
- **New skills** — Propose new research-oriented skills that complement the existing ones.
- **Documentation** — Typo fixes, clearer explanations, additional examples.

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-change`
3. Make your changes
4. If modifying `ai-research-manager/scripts/`, run the tests:
   ```bash
   python -m pytest ai-research-manager/scripts/test_research_index.py
   ```
5. Commit your changes with a clear message
6. Open a pull request

## Skill Structure

Each skill follows this convention:

```
skill-name/
├── SKILL.md          # Required. The main skill file with frontmatter metadata.
├── README.md         # Required. Human-readable documentation.
├── references/       # Optional. Deep reference materials.
├── templates/        # Optional. Document templates.
├── agents/           # Optional. Role/agent specifications.
└── scripts/          # Optional. Validation or utility scripts.
```

### SKILL.md Frontmatter

Every `SKILL.md` must include YAML frontmatter:

```yaml
---
name: skill-name
description: >-
  One-paragraph description of what the skill does and when to trigger it.
---
```

## Guidelines

- **Keep skills tool-agnostic.** Skills should work with any AI assistant that reads markdown, not just one specific IDE.
- **Decision frameworks over raw knowledge.** Skills should teach the AI *when* and *how* to apply techniques, not just list facts.
- **Reference materials should be actionable.** Include code templates, worked examples, and concrete decision rules — not textbook summaries.
- **Test validation scripts.** If your change touches Python scripts, ensure tests pass.
- **Preserve backward compatibility.** Changes to metadata schemas or FSM rules in `ai-research-manager` should be documented and consider migration paths.

## Code of Conduct

Be respectful, constructive, and focused on improving the research experience for everyone.
