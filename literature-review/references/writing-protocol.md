# Writing protocol (strict)

**Principle: the review reports what the evidence base says, at the strength the evidence supports.**
Your own background knowledge is not a source. A statement that cannot be traced to a row in the claims
ledger does not go into the review. When the evidence is thin, say so; do not fill the gap.

The protocol has seven steps. Steps marked **[user]** need the user's confirmation before you continue.

---

## W1. Build the claims ledger before drafting

`06_review/claims.tsv`, tab-separated, one finding per row (start from `templates/claims.tsv`):

| Column | Content |
|---|---|
| `claim_id` | `C001`, `C002`, … unique |
| `paper_id` | `P014`; for a derived claim, every paper it draws on (`P014,P015`) |
| `evidence` | `fulltext` (from your note) · `abstract` (abstract only) · `derived` (your calculation) |
| `location` | `fulltext`: `Fig. 3`, `Table 2, p. 9`, `§4.2`. `abstract`: `abstract`. `derived`: the source claim_ids |
| `statement` | The finding, paraphrased faithfully, **keeping the source's qualifiers and hedging** |
| `numbers` | Every number the review will use from this claim, with units: `0.653 hPa; 0.389 hPa` |
| `section` | Review section(s) it supports |

Illustrative rows (IDs, locations and wording are placeholders; take yours from the notes):

```text
claim_id	paper_id	evidence	location	statement	numbers	section
C001	P0xx	fulltext	Table N, p. N	NA DJF storm track (2–6-day band-pass MSLP s.d. vs ERA5): RMSE 0.653 hPa in CMIP5, 0.389 hPa in CMIP6	0.653 hPa; 0.389 hPa; 2–6 day	§5
C002	P0xx	derived	C001	Relative RMSE reduction CMIP5→CMIP6 = (0.653 − 0.389) / 0.653 = 40%	40%	§5
C003	P0yy	abstract	abstract	Eddy metrics agree in climatological amplitude across 29 CMIP6 models; only upper-level v′v′ correlates with grid spacing	29	§2
```

Rules:
- A `fulltext` claim requires the PDF and `04_notes/<ID>.md`; copy from the note, not from memory.
- Keep the qualifiers that make a number interpretable: models (how many, which), experiment, period,
  season, region, scenario, reference dataset, diagnostic definition.
- Method parameters are claims too (filter band, tracking thresholds, truncation).
- `derived` is only for simple arithmetic on numbers with the **same definition, units, period and reference**
  (e.g. a percentage change within one study). Write the formula in `statement`.
- Never add a row for something you "know". Find it in the evidence base, or propose a manual addition (W6).

## W2. Plan from the ledger **[user]**

- For each section of `templates/review_outline.md`, list the claim_ids it will use.
- A section that rests on one study says so explicitly ("仅一项研究涉及…"); a section with no claims is dropped.
- Show the user the outline with claim_ids and the 3–5 headline findings for the abstract. Draft only after
  they confirm.

## W3. Sentence rules

1. **Cite in the sentence.** Every sentence that states a finding carries its citation, `Author Year（ID）`,
   in that sentence. List items may inherit the citation of a lead-in sentence that ends with a colon.
2. **No uncited prose** except scope statements and transitions, which are marked
   `<!-- no-cite: reason -->` and contain no findings and no numbers.
3. **Numbers exactly as in the ledger.** No rounding, no added precision, no "约/about" unless the source
   says approximately. Always give units and the qualifiers needed to read the number. A number you computed
   is labelled `（本文据…计算）`.
4. **No cross-study statistics.** Do not average, pool or form ranges across studies unless the definitions,
   periods and references are identical, and then only as a labelled derived claim. Otherwise report per study.
5. **Match wording to evidence strength:**

   | Evidence | Use | Never |
   |---|---|---|
   | One study, full text | `X 等（年份，★ID）发现/显示…` · *X et al. found…* | 研究表明（unattributed）, 已证实, 公认 |
   | One study, abstract only | `X 等（年份，ID）报告…`（据摘要）· *X et al. report…* | detail or numbers not in the abstract; causal verbs |
   | ≥2 independent studies agree | `多项研究（A；B）均发现…` · *Several studies (A; B)…* | — |
   | ≥3 independent studies agree, none in the set disagrees | `较为一致/稳健` · *consistent/robust* | 普遍, 公认 without listing them |
   | Studies disagree | `结论不一：A 发现…，而 B 发现…`, plus the reason **if a source gives it** | picking a side; inventing a reason |

   "Independent" means different author groups or different model sets.
6. **Causality.** `导致/驱动/主导/源于` (*cause, drive*) only when the cited study used an attribution design
   (AMIP vs coupled, sensitivity or perturbation experiments, causal discovery) and states the result as causal.
   Otherwise `与…相关/对应` (*associated with*). An interpretation offered by the authors is reported as theirs:
   `作者认为/推测` (*the authors suggest*).
7. **Significance.** `显著/significant` only for statistical significance the source reports (test and level).
   For size, give the number instead of an adjective.
8. **Keep the source's hedging.** *may/might/suggest/likely* → `可能/提示/或许`; never upgrade *may* to `会`, or
   *suggest* to `表明/证明`.
9. **Stay inside the study's scope.** Do not extend a finding beyond its models, region, season, period or
   scenario. Write `N 个 CMIP6 模式` with N, not "CMIP6 模式" for a subset.
10. **No promotional or rhetorical words**: 众所周知、显然、毋庸置疑、首次、突破性、革命性、巨大、严重、彻底、
    完全解决、证明了; *clearly, undoubtedly, for the first time, breakthrough, proves*.
11. **Terminology.** First use gives the English term and abbreviation: `涡动动能（eddy kinetic energy, EKE）`.
    One term per concept throughout.
12. **Grey literature.** Label preprints `（预印本）` and theses `（学位论文）` at first citation. Conference
    abstracts are not cited unless the user approved them as manual additions.
13. **Translate claims, not interpretations.** When a sentence's meaning hinges on a word, keep the original in
    parentheses.

Examples (illustrative; the first uses figures reported in the CMIP6 example review, `P0xx` are placeholders):

| ✗ | ✓ |
|---|---|
| 研究表明，CMIP6 显著改善了北大西洋风暴轴的模拟。 | Harvey 等（2020，★P014）以 2–6 天带通滤波 MSLP 方差衡量，北大西洋冬季（DJF）风暴轴相对 ERA5 的均方根误差由 CMIP5 的 0.653 hPa 降至 CMIP6 的 0.389 hPa。 |
| 海温偏差导致北太平洋风暴轴偏南（P0xx）。 *(abstract-only source)* | 据摘要，X 等（年份，P0xx）报告北太平洋风暴轴偏南与中太平洋冷海温偏差有关。 |
| 未来气旋将普遍增强。 | 在 SSP5-8.5 下，A 等（★P0xx）和 B 等（P0yy）均发现…；C 等（P0zz）则发现…，差异可能来自所用追踪方法（★P0xx 第 4 节）。 |

## W4. Section rules

- **Abstract and Conclusions** restate findings already in the body, with the same citations. No new evidence,
  no new numbers. `check` fails if they cite an ID the body never uses.
- **Introduction**: background statements need citations from the evidence base or approved manual additions.
- **Diagnostics/methods**: parameters as used in the reviewed papers (ledger), not textbook defaults.
- **Mechanisms**: each mechanism is paired with the experiment that supports it; interpretations are labelled as
  the authors'.
- **Uncertainty and outlook** contain only:
  (a) limitations and future work stated by the sources (cited), and
  (b) gaps visible in the evidence base ("本组文献中没有研究评估…"), labelled `本文认为` and citing the claims
  they rest on.
  No research agenda from general knowledge.
- **Synthesis across studies** is labelled (`综合 A、B 的结果，本文认为…`) and cites every study it combines.

## W5. Figures

- Only from ★ papers, after checking the figure against the note.
- Caption describes exactly what the source figure shows: variable, period, season, reference, meaning of
  stippling or hatching, and the source figure number. The takeaway sentence is limited to what the source text says.
- Embed only when the license allows (CC-BY etc., see `download_log.csv`); otherwise describe the figure in text.

## W6. References outside the evidence base

- Never cite from memory.
- If a reference outside the set is genuinely needed (typically the original method paper), ask the user.
- Once approved, add it to `02_metadata/manual_additions.csv`. Columns: `doi, title, authors, year, venue, reason`.
  Verify the DOI resolves to that title first.
- Rerun `curate → select → biblio`. It enters the selected set as tier `M`; record its claims in the ledger like
  any other paper.
- `check` fails on any author–year citation that has no evidence ID.

## W7. Audit loop **[user]**

1. `check` until zero errors.
2. Resolve each warning: overstated wording, `significant` without a test, causal wording resting on abstracts,
   numbers cited only at paragraph level, grey literature, uncited-by-declaration paragraphs.
3. Read the draft sentence by sentence and name the claim_id(s) behind each finding. Delete or rewrite any
   sentence that has none.
4. The user reviews the abstract, the conclusions and every `本文认为` statement before export.

## What `check` enforces and what it cannot

| Enforced (error) | Flagged (warning) | Your responsibility |
|---|---|---|
| Missing claims ledger | `显著/significant` without a reported test | Paraphrase fidelity and hedging (rule 8) |
| Ledger: unknown paper, `fulltext` without PDF/note/location, derived claim without sources | Overstated or promotional words | Scope (rule 9) and evidence-strength wording (rule 5) |
| Cited ID with no ledger claim | Causal wording supported only by abstracts | Causality backed by an attribution design (rule 6) |
| Paragraph without citation (unless `no-cite`) | Numbers cited only at paragraph level | Figure captions faithful to the source |
| Number not in the ledger claims of the cited papers | Paragraphs marked `no-cite` | Disagreements reported, not averaged away |
| Derived number without `本文据…计算` | Counts not found in `stats.json` | Outlook limited to sourced or evidence-visible gaps |
| Consensus wording with <2 citations in the sentence | Cited preprints, theses, abstracts | — |
| Author–year citation without an ID | Selected papers never cited | — |
| Abstract/Conclusions citing evidence absent from the body | Full-text papers without notes | — |
| ★ without full text or on a conference abstract; missing images; IDs outside the set | — | — |
