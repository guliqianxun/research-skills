# Worked example: CMIP6 simulation of synoptic-scale transient activity

The review this skill was distilled from: *CMIP6 模式对天气尺度瞬变活动的模拟能力：研究综述*, written in Chinese.
Its configuration is `templates/topic.example.json`.

## Question and scope

How well do CMIP6 models (including AMIP6, HighResMIP and PAMIP) simulate synoptic-scale transient activity:
storm tracks, extratropical cyclones, transient eddies, EKE, band-pass variance, eddy heat and momentum fluxes,
and baroclinicity? Covers present-day climatology, systematic biases and their causes, improvement over CMIP5,
and future projections. Publication years 2016 onward.

## Funnel

| Step | First run (original scripts) | Rerun with `litreview.py` |
|---|---:|---:|
| Raw records (3 sources × 15 concepts, with duplicates) | 1889 | 1889 |
| After junk filter | — | 1744 |
| Unique (DOI / title) | 1176 | 1176 |
| After near-duplicate merge | (merged after screening) | 1125 |
| Curated (CMIP6 + ≥1 core family, score ≥ 7) | 215 | 194 |
| Tiers A / B / C | 76 / 81 / 58 | carried over by ID, 1 new paper to screen |
| A/B conference abstracts excluded | 0 | 44 |
| Selected core set | 60 (all A) | 52 (35 journal, 14 preprint, 2 thesis, 1 other) |
| Full text obtained | 25 PDFs, 20 counted as read | — (download not rerun) |

The first run's 60 papers included 14 conference abstracts, 12 preprints and 2 theses. Two conference
abstracts were cited as ★ full text. This is why `pub_type`, the exclusion policy and the `check` stage exist.

## Screening rubric used

- **A**: directly evaluates or projects transient activity with CMIP6 output.
- **B**: CMIP6 with transient activity as a secondary diagnostic, or a methods paper applied to CMIP6.
- **C**: other ensembles, storm-resolving models only, regional precipitation only, CMIP6 mentioned in passing.

Example reasons from `tiers.tsv`: `evaluates NH storm track bias across CMIP6 models` (A),
`baroclinic/cyclone secondary; regional orographic focus` (B), `storm-resolving models, not CMIP6` (C).

## Review outline used

1. 引言: why transient activity matters; what CMIP6 changed; the AMIP / HighResMIP / PAMIP experiment families
2. 诊断指标与方法学: diagnostics, see below
3. 气候态模拟评估: NH (DJF, JJA), SH, cyclone intensity and explosive cyclones, consistency between eddy metrics
4. 系统性偏差的驱动机制: SST biases (NH winter), land heating (NH summer), baroclinicity vs static stability,
   moist processes and blocking, resolution dependence
5. 相较 CMIP5 的改进与多模式集合
6. 未来情景下的变化预估: storm-track shifts, cyclone frequency and intensity, cyclone precipitation and jet
   persistence, forced signal vs internal variability, attribution of drivers
7. 不确定性、局限与展望
8. 结论, followed by references (★/☆) and a reproducibility appendix

## Primer: diagnostics of transient activity in CMIP6 studies

Use this to fill the "diagnostics and methods" section and to explain disagreements between papers.

| Paradigm | Diagnostic | Typical settings | Notes |
|---|---|---|---|
| Eulerian | Band-pass variance / standard deviation | 2–6-day filter on daily or 6-hourly MSLP, Z500 or v; DJF/JJA | Works with daily-only output (CMIP3); smooths small systems (e.g. Mediterranean) and loses cyclone identity |
| Eulerian | Eddy fluxes and EKE | v′T′ at 850 hPa, upper-level v′v′, EKE from filtered winds | Metrics agree on climatological amplitude, but sensitivity to resolution differs (upper v′v′ correlates with grid spacing) |
| Lagrangian | Feature tracking (Hodges TRACK) | ξ850 truncated to T42 with wavenumbers ≤5 removed; threshold 1×10⁻⁵ s⁻¹; lifetime ≥48 h; displacement ≥1000 km | Yields track density, genesis/lysis, poleward displacement, peak intensity and deepening rate; results depend on the variable and truncation |
| Baroclinicity | Eady growth rate | σ = 0.31 f \|∂u/∂z\| / N; via thermal wind σ = 0.31 (g/T) \|∂T/∂y\| / N | Separate the temperature-gradient term from the static-stability term N² when attributing bias |
| Extremes | Explosive ("bomb") cyclones | Sanders–Gyakum: central-pressure deepening ≥ 1 bergeron (24 hPa / 24 h × sin φ / sin 60°) | Models generally under-deepen; needs 6-hourly SLP |
| Structure | Cyclone compositing | Rotate and composite around time of maximum intensity | Separates warm and cold conveyor-belt wind structure |
| Emerging | SOM, causal discovery (PCMCI+), EOF-based decompositions | — | Link circulation regimes, teleconnections, and transient vs stationary contributions |

Attribution designs that recur in the literature:
- **AMIP vs coupled** (prescribed vs model SST) separates atmospheric from SST-driven bias.
- **HighResMIP** tests resolution, often confounded with untuned parameterisations.
- **PAMIP** isolates polar-amplification forcing.
- **SSP scenarios** (typically 2070–2100 vs 1979–2014, often SSP5-8.5) with large ensembles to separate the
  forced signal from internal variability.

Reference data: ERA5 is the primary reanalysis; MERRA-2, JRA-55 and NCEP appear as secondary references.
The historical evaluation period is usually 1979–2014.
