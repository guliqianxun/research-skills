#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Systematic literature review pipeline.

    python litreview.py <stage> --project DIR [options]

Stages (run in order; every stage is safe to rerun):
  init      create DIR/topic.json from the template
  search    OpenAlex + Crossref + Semantic Scholar           -> 01_retrieval/{openalex,crossref,s2}.json
  curate    merge, clean, classify type, dedupe, score       -> 02_metadata/papers_raw.csv, papers_curated.csv
  batches   screening batches for A/B/C tiering by the agent -> 02_metadata/screen/batch_NN.txt, RUBRIC.md
  select    merge screen/tiers.tsv and pick the core set     -> 02_metadata/papers_selected.csv
  download  open-access PDFs only                            -> 03_pdfs/<ID>.pdf, download_log.csv, manual_fetch.md
  extract   full text, figure pages, caption index           -> 04_notes/<ID>.fulltext.txt, 05_figures/
  biblio    reference lists, abstract digest, core list      -> 06_review/references*.md, 02_metadata/stats.json
  check     audit a review draft against the evidence files  -> exit code 1 on errors

Configuration: DIR/topic.json (see templates/topic.example.json).
Environment: CONTACT_EMAIL (required for search/download), OPENALEX_API_KEY and S2_API_KEY (optional).
Never put keys in topic.json or in scripts.
"""
import argparse, glob, html, json, os, re, shutil, sys, time, unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from urllib.parse import urljoin

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = ["01_retrieval", "02_metadata", "03_pdfs", "04_notes", "05_figures", "06_review"]
ID_RE = re.compile(r"\bP\d{3,4}\b")
TYPE_RANK = {"journal": 0, "preprint": 1, "thesis": 2, "other": 3, "conference_abstract": 4}


# ---------------------------------------------------------------- project / config

def load_project(root):
    root = os.path.abspath(root)
    p = os.path.join(root, "topic.json")
    if not os.path.exists(p):
        sys.exit(f"missing {p}: run `litreview.py init --project {root}` and edit topic.json")
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    for d in DIRS:
        os.makedirs(os.path.join(root, d), exist_ok=True)
    cfg["_root"] = root
    return cfg


def fp(cfg, *parts):
    return os.path.join(cfg["_root"], *parts)


def read_csv(path):
    if not os.path.exists(path):
        sys.exit(f"missing {path}: run the previous stage first")
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def update_stats(cfg, **kv):
    p = fp(cfg, "02_metadata", "stats.json")
    st = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    st.update(kv)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    return st


def contact_email():
    e = os.environ.get("CONTACT_EMAIL")
    if not e:
        sys.exit("set CONTACT_EMAIL (OpenAlex/Crossref/Unpaywall polite pools need a contact address)")
    return e


# ---------------------------------------------------------------- text helpers

ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5"}


def norm_title(t):
    """Language-neutral title key: keeps CJK, folds accents, 'Part II' == 'Part 2'."""
    t = unicodedata.normalize("NFKD", str(t or "")).lower()
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"\bpart\s+(iii|ii|iv|i|v)\b", lambda m: "part" + ROMAN[m.group(1)], t)
    return re.sub(r"[\W_]+", "", t)


def clean_text(a):
    """Unescape repeatedly (Crossref/OpenAlex abstracts are often double-escaped JATS), then strip tags."""
    a = "" if a is None or (isinstance(a, float) and pd.isna(a)) else str(a)
    for _ in range(3):
        u = html.unescape(a)
        if u == a:
            break
        a = u
    a = re.sub(r"<[^>]+>", " ", a)
    return re.sub(r"\s+", " ", a).strip()


def oa_invert(inv):
    """OpenAlex abstract_inverted_index -> text."""
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


DEFAULT_TYPE_RULES = [
    # (type, regex over "doi | venue | type hints")
    ("conference_abstract", r"10\.5194/egusphere-egu\d{2,4}-|10\.5194/[a-z]+20\d\d-\d+|meeting abstract|fall meeting|general assembly"),
    ("preprint", r"10\.22541/|10\.1002/essoar\.|10\.31223/|10\.21203/|10\.2139/ssrn|10\.48550/|arxiv|"
                 r"10\.5194/egusphere-20\d\d-|10\.5194/[a-z]+-20\d\d-\d+\b|\bpreprint\b|posted-content"),
    ("thesis", r"dissertation|thesis"),
]


def classify(doi, venue, hints, cfg):
    text = " | ".join(str(x).lower() for x in (doi, venue, hints) if isinstance(x, str) and x)
    for typ, rgx in [(k, v) for k, v in cfg.get("pub_type_rules", {}).items()] + DEFAULT_TYPE_RULES:
        if re.search(rgx, text, re.I):
            return typ
    if re.search(r"journal|article", text):
        return "journal"
    return "other"


# ---------------------------------------------------------------- HTTP

def http_headers():
    return {"User-Agent": f"litreview/1.0 (mailto:{contact_email()})"}


def backoff_get(url, params=None, headers=None, tries=6):
    import requests
    h = {**http_headers(), **(headers or {})}
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=h, timeout=45)
        except Exception as e:
            print("   net err", e); time.sleep(3 * (i + 1)); continue
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503):
            try:
                wait = int(r.headers.get("Retry-After", 0))
            except ValueError:
                wait = 0
            wait = wait or 5 * (i + 1)
            print(f"   {r.status_code} backoff {wait}s"); time.sleep(min(wait, 65)); continue
        print("   http", r.status_code, r.text[:120]); return None
    return None


# ---------------------------------------------------------------- stage: init

def stage_init(args):
    root = os.path.abspath(args.project)
    os.makedirs(root, exist_ok=True)
    dst = os.path.join(root, "topic.json")
    if os.path.exists(dst):
        print("exists:", dst)
    else:
        shutil.copy(os.path.join(HERE, "..", "templates", "topic.example.json"), dst)
        print("created:", dst, "-> edit topic, anchor, concepts, scoring and rubric before `search`")
    for d in DIRS:
        os.makedirs(os.path.join(root, d), exist_ok=True)


# ---------------------------------------------------------------- stage: search

def search_openalex(cfg, q, cap):
    q = q.replace(",", " ")  # commas split OpenAlex filters
    params = {"filter": f"title_and_abstract.search:{q},from_publication_date:{cfg['year_min']}-01-01",
              "per-page": 200, "mailto": contact_email()}
    if os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    recs, cursor = [], "*"
    while cursor and len(recs) < cap:
        r = backoff_get("https://api.openalex.org/works", {**params, "cursor": cursor})
        if not r:
            break
        d = r.json()
        recs += d["results"]
        cursor = d["meta"].get("next_cursor")
        if not d["results"]:
            break
        time.sleep(0.3)
    return recs[:cap]


def search_crossref(cfg, q, cap):
    types = ",".join(f"type:{t}" for t in cfg.get("crossref_types", ["journal-article"]))
    recs, cursor = [], "*"
    while cursor and len(recs) < cap:
        r = backoff_get("https://api.crossref.org/works", {
            "query": q, "rows": min(100, cap - len(recs)), "cursor": cursor,
            "filter": f"from-pub-date:{cfg['year_min']}-01-01,{types}",
            "select": "DOI,title,abstract,author,container-title,published,is-referenced-by-count,type",
            "mailto": contact_email()})
        if not r:
            break
        m = r.json()["message"]
        items = m.get("items") or []
        recs += items
        cursor = m.get("next-cursor")
        if not items:
            break
        time.sleep(1.0)
    return recs[:cap]


def search_s2(cfg, q, cap):
    hdr = {"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else None
    recs, off = [], 0
    while len(recs) < cap and off < 1000:
        r = backoff_get("https://api.semanticscholar.org/graph/v1/paper/search", {
            "query": q, "offset": off, "limit": min(100, cap - len(recs)), "year": f"{cfg['year_min']}-",
            "fields": "title,abstract,year,authors,venue,externalIds,citationCount,openAccessPdf,"
                      "isOpenAccess,publicationTypes"}, headers=hdr)
        if not r:
            break
        d = r.json()
        data = d.get("data") or []
        recs += data
        if not data or "next" not in d:
            break
        off = d["next"]
        time.sleep(1.1)
    return recs[:cap]


SEARCHERS = {"openalex": search_openalex, "crossref": search_crossref, "s2": search_s2}


def stage_search(args):
    cfg = load_project(args.project)
    caps = {"openalex": 1000, "crossref": 100, "s2": 100, **cfg.get("max_per_query", {})}
    log = {"date": date.today().isoformat(), "anchor": cfg["anchor"], "year_min": cfg["year_min"], "counts": {}}
    for src in args.sources or cfg.get("sources", list(SEARCHERS)):
        allrecs = {}
        for c in cfg["concepts"]:
            q = f"{cfg['anchor']} {c}".strip()
            print(f"[{src}] {q}")
            try:
                recs = SEARCHERS[src](cfg, q, caps[src])
            except Exception as e:
                print("  FAIL", e); recs = []
            allrecs[c] = recs
            print(f"   -> {len(recs)}")
        with open(fp(cfg, "01_retrieval", f"{src}.json"), "w", encoding="utf-8") as f:
            json.dump(allrecs, f, ensure_ascii=False)
        log["counts"][src] = {c: len(v) for c, v in allrecs.items()}
        print(f"== {src}: {sum(len(v) for v in allrecs.values())} raw records (with duplicates)")
    with open(fp(cfg, "01_retrieval", "search_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- stage: curate

def load_records(cfg):
    rows = []
    p = fp(cfg, "01_retrieval", "openalex.json")
    if os.path.exists(p):
        for recs in json.load(open(p, encoding="utf-8")).values():
            for w in recs:
                src = (w.get("primary_location") or {}).get("source") or {}
                oa = w.get("open_access") or {}
                rows.append(dict(
                    source="openalex", title=w.get("title"), abstract=oa_invert(w.get("abstract_inverted_index")),
                    authors="; ".join(a["author"]["display_name"] for a in (w.get("authorships") or [])[:12]),
                    year=w.get("publication_year"), venue=src.get("display_name"),
                    doi=(w.get("doi") or "").replace("https://doi.org/", "").lower() or None,
                    cited_by_count=w.get("cited_by_count"), is_oa=oa.get("is_oa"), oa_url=oa.get("oa_url"),
                    type_hint=f"{w.get('type') or ''} {src.get('type') or ''}".strip(),
                    retracted=bool(w.get("is_retracted"))))
    p = fp(cfg, "01_retrieval", "crossref.json")
    if os.path.exists(p):
        for recs in json.load(open(p, encoding="utf-8")).values():
            for w in recs:
                dp = (w.get("published") or {}).get("date-parts")
                rows.append(dict(
                    source="crossref", title=" ".join(w.get("title") or []), abstract=w.get("abstract"),
                    authors="; ".join(f"{a.get('given', '')} {a.get('family', '')}".strip()
                                      for a in (w.get("author") or [])[:12]),
                    year=dp[0][0] if dp and dp[0] else None, venue=" ".join(w.get("container-title") or []),
                    doi=(w.get("DOI") or "").lower() or None, cited_by_count=w.get("is-referenced-by-count"),
                    is_oa=None, oa_url=None, type_hint=w.get("type") or "", retracted=False))
    p = fp(cfg, "01_retrieval", "s2.json")
    if os.path.exists(p):
        for recs in json.load(open(p, encoding="utf-8")).values():
            for w in recs:
                ext = w.get("externalIds") or {}
                rows.append(dict(
                    source="s2", title=w.get("title"), abstract=w.get("abstract"),
                    authors="; ".join(a.get("name", "") for a in (w.get("authors") or [])[:12]),
                    year=w.get("year"), venue=w.get("venue"), doi=(ext.get("DOI") or "").lower() or None,
                    cited_by_count=w.get("citationCount"), is_oa=w.get("isOpenAccess"),
                    oa_url=(w.get("openAccessPdf") or {}).get("url") or None,
                    type_hint=" ".join(w.get("publicationTypes") or []), retracted=False))
    # papers the user explicitly approved for citation (e.g. the original method paper outside the
    # search window); they bypass scoring and screening but must carry a reason and a verified DOI
    p = fp(cfg, "02_metadata", "manual_additions.csv")
    if os.path.exists(p):
        for _, m in pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False).iterrows():
            if not m.get("reason"):
                sys.exit(f"manual_additions.csv: '{m.get('title')}' has no reason")
            rows.append(dict(
                source="manual", title=m["title"], abstract=m.get("abstract", ""), authors=m["authors"],
                year=int(m["year"]) if m.get("year") else None, venue=m.get("venue", ""),
                doi=m["doi"].lower() or None, cited_by_count=None, is_oa=None, oa_url=None,
                type_hint="journal-article", retracted=False, manual_reason=m["reason"]))
    return rows


def score_row(row, cfg, rx):
    text = f"{row['title']} {row['abstract']}"
    w = {"anchor": 3, "core": 2, "context": 1, **cfg.get("weights", {})}
    has_anchor = bool(rx["anchor"].search(text))
    core = [k for k, r in rx["core"].items() if r.search(text)]
    ctx = [k for k, r in rx["context"].items() if r.search(text)]
    s = w["anchor"] * has_anchor + w["core"] * len(core) + w["context"] * len(ctx)
    return s, "|".join(core + ctx), has_anchor and len(core) >= 1


def n_cites(x):
    v = pd.to_numeric(x, errors="coerce")
    return 0 if pd.isna(v) else int(v)


def first_nonempty(s):
    return next((x for x in s if isinstance(x, str) and x), None)


def any_oa(s):
    if any(x is True for x in s):
        return True
    return False if any(x is False for x in s) else None


def stage_curate(args):
    cfg = load_project(args.project)
    df = pd.DataFrame(load_records(cfg))
    if df.empty:
        sys.exit("no records in 01_retrieval/: run `search` first")
    n_raw = len(df)
    df["manual_reason"] = df["manual_reason"].fillna("") if "manual_reason" in df else ""
    for c in ("title", "abstract", "venue", "authors"):
        df[c] = df[c].map(clean_text)
    junk = re.compile(cfg.get("junk_title_regex",
                              r"^\s*(?:reply on|response to|comment on|interactive comment|author comment|referee|"
                              r"rc\d|corrigendum|erratum|preface|editorial)\b"), re.I)
    df = df[(df["title"].str.len() > 15) & ~df["title"].str.contains(junk) & ~df["retracted"]]
    n_junk = len(df)

    # exact duplicates: DOI, else title key; keep longest abstract, pool OA/citation info
    df["key"] = df["doi"].fillna("")
    df.loc[df["key"] == "", "key"] = df.loc[df["key"] == "", "title"].map(norm_title)
    df = df[df["key"] != ""]
    df["_alen"] = df["abstract"].str.len()
    df = df.sort_values(["_alen", "source", "key"], ascending=[False, True, True], kind="mergesort")
    agg = df.groupby("key").agg(
        oa_url=("oa_url", first_nonempty), is_oa=("is_oa", any_oa), cited_by_count=("cited_by_count", "max"),
        sources=("source", lambda s: "|".join(sorted(set(s)))),
        type_hint=("type_hint", lambda s: " ".join(sorted({x for x in s if x}))),
        manual_reason=("manual_reason", first_nonempty))
    df = df.drop_duplicates("key").set_index("key")
    for c in agg.columns:
        df[c] = agg[c]
    df = df.reset_index()
    n_unique = len(df)

    df["pub_type"] = [classify(d, v, h, cfg) for d, v, h in zip(df["doi"], df["venue"], df["type_hint"])]

    # near-duplicates (preprint vs published, abstract vs paper, Part I vs Part 1): keep the best version,
    # remember the others so a paywalled article can fall back to its open preprint
    df["_nt"] = df["title"].map(norm_title)
    df["_rank"] = df["pub_type"].map(TYPE_RANK)
    df["_cite"] = pd.to_numeric(df["cited_by_count"], errors="coerce").fillna(0)
    df = df.sort_values(["_rank", "_cite", "key"], ascending=[True, False, True], kind="mergesort")
    alt = df.groupby("_nt").agg(
        other_versions=("doi", lambda s: "|".join(x for x in list(s)[1:] if isinstance(x, str))),
        alt_urls=("oa_url", lambda s: "|".join(x for x in list(s)[1:] if isinstance(x, str) and x)),
        manual_reason=("manual_reason", first_nonempty))
    df = df.drop_duplicates("_nt").set_index("_nt")
    df[["other_versions", "alt_urls", "manual_reason"]] = alt
    df = df.reset_index()
    n_neardup = len(df)

    rx = {"anchor": re.compile(cfg.get("anchor_regex", re.escape(cfg["anchor"])), re.I),
          "core": {k: re.compile(v, re.I) for k, v in cfg["core_groups"].items()},
          "context": {k: re.compile(v, re.I) for k, v in cfg.get("context_tags", {}).items()}}
    df[["relevance_score", "theme_tags", "_core"]] = [score_row(r, cfg, rx) for _, r in df.iterrows()]

    # stable IDs: reuse IDs from a previous papers_raw.csv so screen/tiers.tsv stays valid on reruns
    raw_path = fp(cfg, "02_metadata", "papers_raw.csv")
    known = {}
    if os.path.exists(raw_path):
        old = pd.read_csv(raw_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        for _, r in old.iterrows():  # every DOI and the title key of a work point to its ID
            for k in [r["doi"].lower(), norm_title(r["title"])] + str(r.get("other_versions", "")).lower().split("|"):
                if k:
                    known.setdefault(k, r["id"])
    df = df.sort_values(["relevance_score", "_cite", "key"], ascending=[False, False, True],
                        kind="mergesort").reset_index(drop=True)
    nxt = max([int(i[1:]) for i in known.values()] + [0]) + 1
    ids, used = [], set()
    for _, r in df.iterrows():
        keys = [r["key"], r["_nt"]] + [x.lower() for x in (r["other_versions"] if isinstance(r["other_versions"], str) else "").split("|") if x]
        pid = next((known[k] for k in keys if k in known and known[k] not in used), None)
        if not pid:
            pid, nxt = "P%03d" % nxt, nxt + 1
        used.add(pid)
        ids.append(pid)
    df.insert(0, "id", ids)

    cols = ["id", "title", "authors", "year", "venue", "doi", "pub_type", "cited_by_count", "is_oa", "oa_url",
            "alt_urls", "other_versions", "sources", "relevance_score", "theme_tags", "_core", "manual_reason",
            "abstract"]
    df = df[cols]
    write_csv(df, raw_path)
    min_score = cfg.get("curated_min_score", 7)
    manual = df["manual_reason"].fillna("").astype(str).str.len() > 0
    core = df[(df["_core"] & (df["relevance_score"] >= min_score)) | manual]
    write_csv(core, fp(cfg, "02_metadata", "papers_curated.csv"))
    update_stats(cfg, raw_records=n_raw, after_junk_filter=n_junk, unique_records=n_unique,
                 after_near_duplicates=n_neardup, curated=len(core),
                 curated_by_type=core["pub_type"].value_counts().to_dict())
    print(f"raw {n_raw} -> junk-filtered {n_junk} -> unique {n_unique} -> near-dup {n_neardup} "
          f"-> curated {len(core)} (score>={min_score}, anchor + >=1 core concept)")
    print("curated by type:", core["pub_type"].value_counts().to_dict())
    cnt = Counter(t for tags in core["theme_tags"] for t in str(tags).split("|") if t)
    print("theme tags:", dict(cnt.most_common()))


# ---------------------------------------------------------------- stage: batches / select

def read_tiers(cfg):
    p = fp(cfg, "02_metadata", "screen", "tiers.tsv")
    tiers = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and ID_RE.fullmatch(parts[0].strip()) and parts[1].strip() in ("A", "B", "C"):
                    tiers[parts[0].strip()] = (parts[1].strip(), parts[2].strip() if len(parts) > 2 else "")
    return tiers


def stage_batches(args):
    cfg = load_project(args.project)
    sdir = fp(cfg, "02_metadata", "screen")
    os.makedirs(sdir, exist_ok=True)
    cur = read_csv(fp(cfg, "02_metadata", "papers_curated.csv"))
    done = read_tiers(cfg)
    manual = cur["manual_reason"].fillna("").astype(str).str.len() > 0 if "manual_reason" in cur else cur["id"].isna()
    todo = cur[~cur["id"].isin(done) & ~manual]
    scr = cfg.get("screen", {})
    rubric = scr.get("rubric", {})
    with open(os.path.join(sdir, "RUBRIC.md"), "w", encoding="utf-8") as f:
        f.write(f"# Screening rubric: {cfg['topic']}\n\n")
        for t in ("A", "B", "C"):
            f.write(f"- **{t}**: {rubric.get(t, '')}\n")
        f.write("\nJudge from title + abstract only. Append one line per paper to `tiers.tsv`:\n\n"
                "    <ID>\\t<A|B|C>\\t<reason, <= 15 words>\n\n"
                "Do not skip papers. TYPE is informative: a conference abstract can be relevant (tier it on "
                "content) but `select` excludes types listed in select.exclude_types.\n")
    existing = [int(m.group(1)) for f in os.listdir(sdir) if (m := re.match(r"batch_(\d+)\.txt$", f))]
    n0 = max(existing) + 1 if existing else 0
    size = scr.get("batch_size", 45)
    files = []
    for b, i in enumerate(range(0, len(todo), size)):
        chunk = todo.iloc[i:i + size]
        path = os.path.join(sdir, f"batch_{n0 + b:02d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for _, r in chunk.iterrows():
                cites = n_cites(r["cited_by_count"])
                f.write(f"ID: {r['id']}\nTITLE: {r['title']}\n"
                        f"YEAR: {r['year']} | CITED: {cites} | OA: {r['is_oa']} | TYPE: {r['pub_type']} | "
                        f"TAGS: {r['theme_tags']}\nABSTRACT: {str(r['abstract'])[:1500]}\n---\n")
        files.append(os.path.basename(path))
    print(f"{len(cur) - len(todo) - int(manual.sum())} of {len(cur)} curated already tiered, {int(manual.sum())} manual; {len(todo)} to screen in {len(files)} batch(es): {', '.join(files) or '-'}")


def stage_select(args):
    cfg = load_project(args.project)
    sc = {"target_min": 45, "max": 60, "exclude_types": ["conference_abstract"], **cfg.get("select", {})}
    df = read_csv(fp(cfg, "02_metadata", "papers_curated.csv"))
    tiers = read_tiers(cfg)
    manual = df["manual_reason"].fillna("").astype(str).str.len() > 0 if "manual_reason" in df else df["id"].isna()
    missing = sorted(set(df.loc[~manual, "id"]) - set(tiers))
    if missing and not args.allow_untiered:
        sys.exit(f"{len(missing)} curated papers have no tier (e.g. {', '.join(missing[:8])}). "
                 f"Run `batches`, screen them, or pass --allow-untiered to treat them as C.")
    df["tier"] = df["id"].map(lambda i: tiers.get(i, ("C", ""))[0])
    df["screen_reason"] = df["id"].map(lambda i: tiers.get(i, ("", "untiered"))[1])
    df.loc[manual, "tier"] = "M"
    df.loc[manual, "screen_reason"] = "manual addition: " + df.loc[manual, "manual_reason"].astype(str)
    tier_counts = df["tier"].value_counts().to_dict()
    pool = df[~df["pub_type"].isin(sc["exclude_types"]) & ~manual].copy()
    pool["_cite"] = pd.to_numeric(pool["cited_by_count"], errors="coerce").fillna(0)
    pool = pool.sort_values(["relevance_score", "_cite"], ascending=False)
    sel = pool[pool["tier"] == "A"]
    if len(sel) < sc["target_min"]:
        sel = pd.concat([sel, pool[pool["tier"] == "B"].head(sc["target_min"] - len(sel))])
    sel = pd.concat([sel.head(sc["max"]).drop(columns="_cite"), df[manual]]).reset_index(drop=True)
    write_csv(sel, fp(cfg, "02_metadata", "papers_selected.csv"))
    excluded = df[df["pub_type"].isin(sc["exclude_types"]) & df["tier"].isin(["A", "B"])]
    update_stats(cfg, tiers=tier_counts, selected=len(sel), selected_by_type=sel["pub_type"].value_counts().to_dict(),
                 selected_by_tier=sel["tier"].value_counts().to_dict(), excluded_by_type_AB=len(excluded),
                 manual_additions=int(manual.sum()))
    print("tiers:", tier_counts)
    print(f"selected {len(sel)} | by tier {sel['tier'].value_counts().to_dict()} | "
          f"by type {sel['pub_type'].value_counts().to_dict()}")
    if len(excluded):
        print(f"excluded {len(excluded)} A/B papers of type {sc['exclude_types']}: {', '.join(excluded['id'][:15])}")
    pre = sel[sel["pub_type"] == "preprint"]
    if len(pre):
        print(f"{len(pre)} preprints selected; check for a published version: {', '.join(pre['id'])}")


# ---------------------------------------------------------------- stage: download

PDF_META = [re.compile(rb'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']', re.I),
            re.compile(rb'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']', re.I)]
PDF_LINK = [re.compile(rb'href=["\']([^"\']+?\.pdf(?:\?[^"\']*)?)["\']', re.I),
            re.compile(rb'href=["\']([^"\']*(?:/pdf/|pdfdirect|/content/pdf)[^"\']*)["\']', re.I)]


def valid_pdf(path):
    if not os.path.exists(path) or os.path.getsize(path) < 20000:
        return False
    with open(path, "rb") as f:
        return f.read(5) == b"%PDF-"


def fetch(url):
    """-> ('pdf', bytes) | ('html', (bytes, final_url)) | ('err', msg)"""
    import requests
    for i in range(2):
        try:
            r = requests.get(url, headers={**http_headers(), "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8"},
                             timeout=(15, 60), allow_redirects=True)
        except Exception as e:
            if i:
                return "err", f"neterr:{type(e).__name__}"
            time.sleep(2); continue
        if r.status_code != 200:
            return "err", f"http{r.status_code}"
        if r.content[:5] == b"%PDF-":
            return "pdf", r.content
        return "html", (r.content, r.url)
    return "err", "retries"


def pdf_from_html(data, base):
    for rgx in PDF_META + PDF_LINK:
        m = rgx.search(data)
        if m:
            u = m.group(1).decode("utf-8", "ignore")
            return "https:" + u if u.startswith("//") else urljoin(base, u)
    return None


def candidates(r):
    """Ordered OA candidate URLs with their origin and license."""
    import requests
    doi = r["doi"] if isinstance(r["doi"], str) else None
    out, lic = [], ""
    if doi:
        try:
            u = requests.get(f"https://api.unpaywall.org/v2/{doi}", params={"email": contact_email()}, timeout=30)
            if u.status_code == 200:
                d = u.json()
                lic = ((d.get("best_oa_location") or {}).get("license")) or ""
                for loc in [d.get("best_oa_location")] + (d.get("oa_locations") or []):
                    if loc and loc.get("url_for_pdf"):
                        out.append(("unpaywall", loc["url_for_pdf"]))
                for loc in [d.get("best_oa_location")] + (d.get("oa_locations") or []):
                    if loc and loc.get("url"):
                        out.append(("unpaywall-landing", loc["url"]))
        except Exception:
            pass
    if isinstance(r.get("oa_url"), str):
        out.insert(0, ("oa_url", r["oa_url"]))
    if doi:
        try:
            s = requests.get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
                             params={"fields": "openAccessPdf"}, timeout=30)
            if s.status_code == 200 and (s.json().get("openAccessPdf") or {}).get("url"):
                out.append(("s2", s.json()["openAccessPdf"]["url"]))
        except Exception:
            pass
        oa = r.get("oa_url") if isinstance(r.get("oa_url"), str) else ""
        if "copernicus.org" in oa and not oa.endswith(".pdf"):
            out.append(("pattern", oa.rstrip("/") + ".pdf"))
        if "mdpi.com" in oa:
            out.append(("pattern", oa.rstrip("/") + "/pdf"))
        if doi.startswith(("10.1029/", "10.1002/")) and r.get("is_oa") is True:
            out.append(("pattern", f"https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/{doi}"))
        if doi.startswith("10.1007/") and r.get("is_oa") is True:
            out.append(("pattern", f"https://link.springer.com/content/pdf/{doi}.pdf"))
    for u in str(r.get("alt_urls") or "").split("|"):
        if u.startswith("http"):
            out.append(("other-version", u))
    seen, uniq = set(), []
    for src, u in out:
        if u not in seen:
            seen.add(u); uniq.append((src, u))
    return uniq, lic


def download_one(cfg, r):
    pid = r["id"]
    dest = fp(cfg, "03_pdfs", f"{pid}.pdf")
    if valid_pdf(dest):
        return dict(id=pid, status="cached", source="", url="", license="")
    cands, lic = candidates(r)
    for src, url in cands:
        kind, payload = fetch(url)
        if kind == "html":
            nxt = pdf_from_html(*payload)
            kind, payload = fetch(nxt) if nxt and nxt != url else ("err", "html-without-pdf-link")
        if kind == "pdf" and len(payload) >= 20000:
            with open(dest, "wb") as f:
                f.write(payload)
            print(f"{pid} ok [{src}] {len(payload) // 1024} KB")
            return dict(id=pid, status="ok", source=src, url=url, license=lic)
        print(f"{pid} [{src}] {payload if isinstance(payload, str) else 'not a pdf'}")
    return dict(id=pid, status="FAIL", source="", url=str(r["doi"]), license=lic)


def stage_download(args):
    cfg = load_project(args.project)
    sel = read_csv(fp(cfg, "02_metadata", "papers_selected.csv"))
    if args.ids:
        sel = sel[sel["id"].isin(args.ids)]
    rows = [r for _, r in sel.iterrows()]
    log = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(download_one, cfg, r): r["id"] for r in rows}
        for fu in as_completed(futs):
            try:
                log.append(fu.result())
            except Exception as e:
                log.append(dict(id=futs[fu], status="FAIL", source="", url=f"exc:{e}", license=""))
    ld = pd.DataFrame(log).sort_values("id")
    write_csv(ld, fp(cfg, "02_metadata", "download_log.csv"))
    fails = sel[sel["id"].isin(ld.loc[ld["status"] == "FAIL", "id"])]
    with open(fp(cfg, "02_metadata", "manual_fetch.md"), "w", encoding="utf-8") as f:
        f.write(f"# Papers to fetch by hand ({len(fails)})\n\nGet them through your library, save as "
                f"`03_pdfs/<ID>.pdf`, then rerun `extract` and `biblio`.\n\n")
        for _, r in fails.iterrows():
            link = f"https://doi.org/{r['doi']}" if isinstance(r["doi"], str) else "(no DOI)"
            f.write(f"- [ ] **{r['id']}** {link} — {r['title']} ({r['pub_type']})\n")
    ok = ld["status"].isin(["ok", "cached"]).sum()
    print(f"\n== {ok}/{len(ld)} PDFs available; {len(fails)} listed in 02_metadata/manual_fetch.md")


# ---------------------------------------------------------------- stage: extract

CAPTION = re.compile(r"^\s*(?:Fig(?:ure)?\.?|FIG\.?|图)\s*(\d+)\s*[.:：|]?\s*(.*)", re.I)


def stage_extract(args):
    cfg = load_project(args.project)
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz  # PyMuPDF < 1.24
        except ImportError:
            sys.exit("extract needs PyMuPDF: pip install pymupdf")
    ex = {"dpi": 130, "max_pages_per_paper": 12, **cfg.get("extract", {})}
    pdfs = sorted(glob.glob(fp(cfg, "03_pdfs", "*.pdf")))
    if args.ids:
        pdfs = [p for p in pdfs if os.path.basename(p)[:-4] in args.ids]
    idx_path = fp(cfg, "05_figures", "figures_index.csv")
    index = read_csv(idx_path).to_dict("records") if os.path.exists(idx_path) else []
    for pdf in pdfs:
        pid = os.path.basename(pdf)[:-4]
        index = [x for x in index if x["id"] != pid]
        try:
            doc = fitz.open(pdf)
        except Exception as e:
            print(pid, "OPEN FAIL", e); continue
        pages = [pg.get_text() for pg in doc]
        with open(fp(cfg, "04_notes", f"{pid}.fulltext.txt"), "w", encoding="utf-8") as f:
            f.write("".join(f"\n\n=== page {i + 1} ===\n{t}" for i, t in enumerate(pages)))
        saved = 0
        for i, pg in enumerate(doc):
            caps = [m for m in (CAPTION.match(ln) for ln in pages[i].splitlines()) if m]
            graphic = pg.get_images() or len(pg.get_drawings()) > 50
            if not caps or not graphic or saved >= ex["max_pages_per_paper"]:
                continue
            name = f"{pid}_p{i + 1:02d}.png"
            pg.get_pixmap(dpi=ex["dpi"]).save(fp(cfg, "05_figures", name))
            saved += 1
            for m in caps:
                index.append(dict(id=pid, page=i + 1, figure=int(m.group(1)), caption=m.group(2)[:200], file=name))
        print(f"{pid}: {len(doc)} pages, {sum(map(len, pages))} chars, {saved} figure pages")
        doc.close()
    if index:
        write_csv(pd.DataFrame(index).sort_values(["id", "page", "figure"]), idx_path)


# ---------------------------------------------------------------- stage: biblio

def first_author(a):
    a = str(a)
    return "Anon" if not a or a == "nan" else a.split(";")[0].strip()


def cite_line(r):
    star = "★" if r["has_fulltext"] else "☆"
    yr = int(r["year"]) if pd.notna(r["year"]) else "n.d."
    venue = r["venue"] if isinstance(r["venue"], str) and r["venue"] else f"({r['pub_type']})"
    if r["pub_type"] in ("preprint", "conference_abstract", "thesis") and isinstance(r["venue"], str) and r["venue"]:
        venue += f" ({r['pub_type']})"
    doi = f"https://doi.org/{r['doi']}" if isinstance(r["doi"], str) else "(no DOI)"
    etal = " et al." if ";" in str(r["authors"]) else ""
    return star, f"{first_author(r['authors'])}{etal} ({yr}). {r['title']}. *{venue}*. {doi}"


HEADERS = {
    "zh": {"id": "编号", "title": "标题", "authors": "第一作者等", "year": "年份", "venue": "期刊", "doi": "DOI",
           "pub_type": "文献类型", "cited_by_count": "被引次数", "is_oa": "开放获取", "has_fulltext": "是否精读全文",
           "theme_tags": "主题标签", "relevance_score": "相关性得分", "screen_reason": "筛选说明", "abstract": "摘要"},
}


def stage_biblio(args):
    cfg = load_project(args.project)
    sel = read_csv(fp(cfg, "02_metadata", "papers_selected.csv"))
    sel["has_fulltext"] = [valid_pdf(fp(cfg, "03_pdfs", f"{i}.pdf")) and t != "conference_abstract"
                           for i, t in zip(sel["id"], sel["pub_type"])]
    write_csv(sel, fp(cfg, "02_metadata", "papers_selected.csv"))
    n, nf = len(sel), int(sel["has_fulltext"].sum())

    lines = [f"# Reference list by ID ({n} papers; ★ full text {nf}, ☆ title + abstract {n - nf})\n"]
    for _, r in sel.sort_values("id").iterrows():
        star, c = cite_line(r)
        lines.append(f"- {star} **{r['id']}** {c} [cited {n_cites(r['cited_by_count'])}]")
    with open(fp(cfg, "06_review", "references.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    sel["_surname"] = sel["authors"].map(lambda a: first_author(a).split()[-1].lower() if first_author(a) else "")
    lines = [f"## References\n\n> {n} papers, sorted by first-author surname; ★ = full text read ({nf}), "
             f"☆ = title + abstract only ({n - nf}). [Pxxx] is the internal ID used in notes and figure captions.\n"]
    for k, (_, r) in enumerate(sel.sort_values(["_surname", "year"]).iterrows(), 1):
        star, c = cite_line(r)
        lines.append(f"{k}. {star} {c}  [{r['id']}]")
    with open(fp(cfg, "06_review", "references_by_author.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    nfdf = sel[~sel["has_fulltext"]].sort_values(["relevance_score", "cited_by_count"], ascending=False)
    lines = [f"# Abstract digest: {len(nfdf)} papers without full text\n"]
    for _, r in nfdf.iterrows():
        lines.append(f"\n## {r['id']} | {r['year']} | {r['pub_type']} | tags: {r['theme_tags']}\n"
                     f"**{r['title']}**\nDOI: {r['doi']}\n{str(r['abstract'])[:900]}")
    with open(fp(cfg, "02_metadata", "abstracts_digest.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    lang = cfg.get("language", "en")
    cols = list(HEADERS["zh"])
    core = sel[[c for c in cols if c in sel.columns]].copy()
    core["authors"] = core["authors"].map(lambda a: first_author(a) + (" et al." if ";" in str(a) else ""))
    if lang in HEADERS:
        core = core.rename(columns=HEADERS[lang])
    stem = fp(cfg, "core_papers")
    write_csv(core, stem + ".csv")
    try:
        core.to_excel(stem + ".xlsx", index=False)
    except ImportError:
        print("openpyxl not installed: skipped core_papers.xlsx")

    no_notes = [i for i, f in zip(sel["id"], sel["has_fulltext"]) if f and not os.path.exists(fp(cfg, "04_notes", f"{i}.md"))]
    st = update_stats(cfg, selected=n, fulltext=nf, abstract_only=n - nf,
                      selected_by_type=sel["pub_type"].value_counts().to_dict(), biblio_date=date.today().isoformat())
    print(f"{n} selected: {nf} full text, {n - nf} title+abstract -> references.md, references_by_author.md, "
          f"abstracts_digest.md, core_papers.csv/.xlsx")
    if no_notes:
        print(f"WARNING {len(no_notes)} full-text papers have no reading notes (04_notes/<ID>.md): {', '.join(no_notes)}")
    print("stats:", json.dumps(st, ensure_ascii=False))


# ---------------------------------------------------------------- stage: check

COUNT_RE = re.compile(r"(\d+)\s*(?:篇|papers?\b|articles?\b|studies\b)", re.I)
# a number followed by a physical unit, percent or count word: these must be traceable to the ledger
QTY_RE = re.compile(r"(?<![\w.])[-−+±]?(\d+(?:\.\d+)?)(?=\s*(?:%|％|‰|hPa|Pa\b|°|℃|K\b|km|m\s*s|m/s|mm|cm|kg|W\s*m|"
                    r"天|日|小时|h\b|个|倍|SYPD|bergeron))")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
NO_CITE = re.compile(r"<!--\s*no-cite:\s*(.*?)\s*-->")
LOCATION_RE = re.compile(r"Fig|Table|\bp\.|pp\.|page|Sec|§|Eq|图|表|页|节|式", re.I)
DERIVED_LABEL = re.compile(r"本文(?:据|计算|估算|统计)|本综述(?:据|计算|统计)|computed (?:here|in this review)|"
                           r"our calculation|this review'?s calculation", re.I)
AUTHOR_YEAR = re.compile(r"\b[A-Z][a-z]+(?:[-'][A-Za-z]+)*(?:\s*(?:et al\.?|等|&|and)\s*(?:[A-Z][a-z]+)?)?"
                         r"\s*[（(,，]?\s*(?:19|20)\d{2}[a-z]?\b")
CONSENSUS = re.compile(r"多项研究|多数研究|大量研究|诸多研究|众多研究|普遍认为|广泛认为|研究普遍|学界共识|"
                       r"several studies|many studies|most studies|numerous studies|widely (?:accepted|agreed)|"
                       r"there is (?:a )?consensus", re.I)
RISKY = re.compile(r"众所周知|显而易见|显然|毋庸置疑|无疑|必然|首次|突破性|革命性|彻底|完全解决|证明了|证实了|"
                   r"it is well known|\bclearly\b|undoubtedly|obviously|\bproves?\b|for the first time|"
                   r"groundbreaking|breakthrough", re.I)
SIGNIF = re.compile(r"显著|significant", re.I)
SIGNIF_OK = re.compile(r"p\s*[<≤]|置信|信度|检验|confidence|significance level|t-test|\btest\b", re.I)
# the reviewer's own judgement is not allowed: report what the sources say, attributed
REVIEWER_VIEW = re.compile(r"本文认为|本综述认为|笔者认为|我们认为|本文推测|本文判断|我们推测|本文建议|我们建议|"
                           r"in (?:our|this review'?s) (?:view|assessment|opinion)|"
                           r"\bwe (?:argue|believe|suggest|propose|recommend|speculate|conclude)\b|"
                           r"this review (?:argues|suggests|proposes|recommends|concludes)", re.I)
# inference connectives: allowed only when the sentence cites the paper that drew the inference
INFERENCE = re.compile(r"由此可见|可以推断|据此推断|不难看出|这意味着|这说明|这表明|"
                       r"this (?:implies|suggests|indicates|means)|it follows that", re.I)
CAUSAL = re.compile(r"导致|引起|驱动|主导|决定了|造成|源于|归因于|\bcause[sd]?\b|\bdrive[ns]?\b|driven by|"
                    r"\bleads? to\b|attribut", re.I)


def review_units(text, skip_rx):
    """-> [(section, text, inherited_ids)] for every unit that must carry a citation: paragraph, list item,
    table row, caption. List items inherit the IDs of a lead-in paragraph that ends with a colon.
    Headings, blockquotes (scope box), images and skipped sections (references, appendix) are left out."""
    text = re.sub(r"<!--(?!\s*no-cite:).*?-->", "", text, flags=re.S)  # template guidance, not content
    units, cur, section, skip, table_row = [], [], "", False, 0
    state = {"lead_ids": set()}

    def flush(is_item):
        if cur and not skip:
            t = " ".join(cur)
            inherited = state["lead_ids"] if is_item else set()
            units.append((section, t, inherited))
            if not is_item:
                state["lead_ids"] = set(ID_RE.findall(t)) if re.search(r"[:：]\s*$", t) else set()
        cur.clear()

    item = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            flush(item); item = False; state["lead_ids"] = set()
            section = s.lstrip("#").strip(); skip = bool(skip_rx.search(section)); continue
        if s.startswith("|"):
            flush(item); item = False; table_row += 1
            if table_row > 2 and not skip:  # header and separator rows carry no claims
                units.append((section, s, set()))
            continue
        table_row = 0
        if not s or re.fullmatch(r"[-*_]{3,}", s) or s.startswith((">", "![")):
            flush(item); continue
        if re.match(r"([-*+]|\d+[.)、])\s", s):
            flush(item); item = True
        elif not cur:
            item = False
        cur.append(s)
    flush(item)
    return units


def split_sentences(t):
    t = t.replace("et al.", "et al§")
    parts = re.split(r"(?<=[。！？；])|(?<=[.!?;])\s+(?=[A-Z（(])", t)
    return [p.replace("et al§", "et al.").strip() for p in parts if p and p.strip()]


def numbers_in(s):
    return {float(x) for x in NUM_RE.findall(s or "")}


def load_ledger(cfg, info, errors, warns):
    """Validate 06_review/claims.tsv -> ({pid: set(numbers)}, {pid: set(derived numbers)}, {pid: n claims})."""
    import csv
    p = fp(cfg, "06_review", "claims.tsv")
    if not os.path.exists(p):
        return None
    led = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig", quoting=csv.QUOTE_NONE)
    need = {"claim_id", "paper_id", "evidence", "location", "statement", "numbers"}
    if need - set(led.columns):
        errors.append(f"claims.tsv is missing columns: {', '.join(sorted(need - set(led.columns)))}")
        return None
    direct, derived, n_claims = {}, {}, Counter()
    ids = set(led["claim_id"])
    for dup in sorted(led.loc[led["claim_id"].duplicated(), "claim_id"]):
        errors.append(f"claims.tsv: duplicate claim_id {dup}")
    for _, c in led.iterrows():
        cid, ev, loc = c["claim_id"], c["evidence"].strip(), c["location"].strip()
        pids = [x.strip() for x in re.split(r"[,;，\s]+", c["paper_id"]) if x.strip()]
        if not c["statement"].strip() or not pids:
            errors.append(f"claims.tsv {cid}: empty statement or paper_id")
            continue
        if ev not in ("fulltext", "abstract", "derived"):
            errors.append(f"claims.tsv {cid}: evidence must be fulltext | abstract | derived, not '{ev}'")
        nums = numbers_in(c["numbers"]) | numbers_in(c["statement"])
        for pid in pids:
            if pid not in info.index:
                errors.append(f"claims.tsv {cid}: {pid} is not in papers_selected.csv")
                continue
            n_claims[pid] += 1
            if ev == "fulltext":
                if not bool(info.at[pid, "has_fulltext"]):
                    errors.append(f"claims.tsv {cid}: evidence=fulltext but {pid} has no full text")
                elif not os.path.exists(fp(cfg, "04_notes", f"{pid}.md")):
                    errors.append(f"claims.tsv {cid}: evidence=fulltext but 04_notes/{pid}.md does not exist")
            (derived if ev == "derived" else direct).setdefault(pid, set()).update(nums)
        if ev == "fulltext" and not LOCATION_RE.search(loc):
            errors.append(f"claims.tsv {cid}: full-text claim needs a location (Fig./Table/p./§), got '{loc}'")
        if ev == "abstract" and loc.lower() not in ("abstract", "摘要"):
            warns.append(f"claims.tsv {cid}: evidence=abstract but location is '{loc}'")
        if ev == "derived":
            src = [x for x in re.split(r"[,;，\s]+", loc) if x]
            bad = [x for x in src if x not in ids]
            if not src or bad:
                errors.append(f"claims.tsv {cid}: derived claim must list its source claim_ids in location "
                              f"(unknown: {', '.join(bad) or 'none given'})")
    return direct, derived, n_claims


def stage_check(args):
    cfg = load_project(args.project)
    wcfg = {"require_ledger": True, "skip_headings": r"参考文献|References|附[:：录]|Appendix|致谢|Acknowledg",
            **cfg.get("writing", {})}
    review = args.review
    if not review:
        mds = [p for p in glob.glob(fp(cfg, "06_review", "*.md"))
               if not os.path.basename(p).startswith(("references", "abstracts"))]
        if not mds:
            sys.exit("no review draft in 06_review/: pass --review PATH")
        review = max(mds, key=os.path.getmtime)
    text = open(review, encoding="utf-8").read()
    sel = read_csv(fp(cfg, "02_metadata", "papers_selected.csv"))
    if "has_fulltext" not in sel.columns:
        sys.exit("papers_selected.csv has no has_fulltext column: run `biblio` first")
    if "pub_type" not in sel.columns:
        sel["pub_type"] = [classify(d, v, "", cfg) for d, v in zip(sel["doi"], sel["venue"])]
    info = sel.set_index("id")
    has_ft = lambda pid: pid in info.index and bool(info.at[pid, "has_fulltext"])
    errors, warns = [], []

    # --- evidence base: IDs, ★ claims, images, grey literature, counts
    starred = set(re.findall(r"★\s*(P\d{3,4})\b", text))
    cited = set(ID_RE.findall(text))
    for pid in sorted(cited - set(info.index)):
        errors.append(f"{pid} is cited but not in papers_selected.csv")
    for pid in sorted(starred & set(info.index)):
        if not has_ft(pid):
            errors.append(f"★{pid} claims full-text reading, but there is no full text")
        if info.at[pid, "pub_type"] == "conference_abstract":
            errors.append(f"★{pid} is a conference abstract: there is no full paper to have read")
    for pid in sorted(set(info.index) - cited):
        warns.append(f"{pid} is selected but never cited")
    for pid in sorted(info.index[info["has_fulltext"].astype(bool)]):
        if not os.path.exists(fp(cfg, "04_notes", f"{pid}.md")):
            warns.append(f"{pid} has full text but no reading notes 04_notes/{pid}.md")
    for target in re.findall(r"!\[[^\]]*\]\(([^)\s]+)", text):
        if not target.startswith("http") and not os.path.exists(os.path.join(os.path.dirname(review), target)):
            errors.append(f"image not found: {target}")
    grey = sorted(pid for pid in cited & set(info.index)
                  if info.at[pid, "pub_type"] in ("conference_abstract", "preprint", "thesis"))
    if grey:
        warns.append(f"{len(grey)} cited items are not peer-reviewed articles (label them in the text): {', '.join(grey)}")
    stats_path = fp(cfg, "02_metadata", "stats.json")
    actual = {"selected": len(sel), "fulltext": int(sel["has_fulltext"].sum()),
              "abstract_only": int((~sel["has_fulltext"].astype(bool)).sum())}
    if os.path.exists(stats_path):
        st = json.load(open(stats_path, encoding="utf-8"))
        known = {v for v in st.values() if isinstance(v, int)} | set(actual.values())
        for k in ("tiers", "selected_by_type", "selected_by_tier"):
            known |= {v for v in st.get(k, {}).values() if isinstance(v, int)}
        for m in COUNT_RE.finditer(text):
            if int(m.group(1)) not in known:
                ctx = text[max(0, m.start() - 25):m.end() + 5].replace("\n", " ")
                warns.append(f"count '{m.group(0)}' matches no value in stats.json: …{ctx}…")

    # --- claims ledger
    ledger = load_ledger(cfg, info, errors, warns)
    if ledger is None and wcfg["require_ledger"]:
        errors.append("no claims ledger 06_review/claims.tsv: every claim must be recorded there before writing")
    direct, derived, n_claims = ledger or ({}, {}, Counter())
    if ledger is not None:
        for pid in sorted(cited & set(info.index)):
            if not n_claims[pid]:
                errors.append(f"{pid} is cited in the review but has no claim in claims.tsv")

    # --- writing rules, unit by unit and sentence by sentence
    skip_rx = re.compile(wcfg["skip_headings"], re.I)
    body_ids, summary_ids = set(), {}
    for section, unit, inherited in review_units(text, skip_rx):
        own_ids = set(ID_RE.findall(unit))
        unit_ids = own_ids or inherited
        is_summary = bool(re.search(r"摘要|Abstract|结论|Conclusion", section, re.I))
        if is_summary:
            summary_ids.setdefault(section, set()).update(own_ids)
        else:
            body_ids |= own_ids
        short = re.sub(r"\s+", " ", NO_CITE.sub("", unit))[:60]
        if not unit_ids:
            m = NO_CITE.search(unit)
            if m:
                warns.append(f"[{section}] uncited by declaration ({m.group(1)}): {short}…")
            else:
                errors.append(f"[{section}] paragraph has no citation (cite it or mark <!-- no-cite: reason -->): {short}…")
        for sent in split_sentences(NO_CITE.sub("", unit)):
            ids_s = set(ID_RE.findall(sent))
            ctx_ids = ids_s or unit_ids
            snippet = sent[:70]
            if not ids_s and AUTHOR_YEAR.search(sent):
                errors.append(f"[{section}] author-year citation without an evidence ID (outside the evidence base?): {snippet}")
            if CONSENSUS.search(sent) and len(ids_s) < 2:
                errors.append(f"[{section}] consensus wording needs >=2 cited studies in the same sentence: {snippet}")
            if m := RISKY.search(sent):
                warns.append(f"[{section}] overstated wording '{m.group(0)}': {snippet}")
            if SIGNIF.search(sent) and not SIGNIF_OK.search(sent):
                errors.append(f"[{section}] '显著/significant' without the reported test in the sentence (e.g. 95% 置信水平, "
                              f"t 检验); otherwise give the number instead: {snippet}")
            if m := REVIEWER_VIEW.search(sent):
                errors.append(f"[{section}] reviewer's own judgement '{m.group(0)}' is not allowed; report the sources' "
                              f"statements, attributed: {snippet}")
            if (m := INFERENCE.search(sent)) and not ids_s:
                errors.append(f"[{section}] inference '{m.group(0)}' without the citation of the paper that drew it: {snippet}")
            if CAUSAL.search(sent) and ctx_ids and not any(has_ft(p) for p in ctx_ids):
                warns.append(f"[{section}] causal wording resting on abstracts only: {snippet}")
            for q in QTY_RE.finditer(sent):
                val = float(q.group(1))
                if not ctx_ids:
                    errors.append(f"[{section}] number '{q.group(0)}' has no citation: {snippet}")
                    continue
                if ledger is None:
                    continue
                if any(val in direct.get(p, ()) for p in ctx_ids):
                    if not ids_s:
                        warns.append(f"[{section}] number '{q.group(0)}' is cited only at paragraph level: {snippet}")
                elif any(val in derived.get(p, ()) for p in ctx_ids):
                    if not DERIVED_LABEL.search(sent):
                        errors.append(f"[{section}] derived number '{q.group(0)}' must be labelled as this review's "
                                      f"calculation (e.g. 本文据…计算): {snippet}")
                else:
                    errors.append(f"[{section}] number '{q.group(0)}' is not in the claims of {', '.join(sorted(ctx_ids))}: {snippet}")
    for section, ids in summary_ids.items():
        new = sorted(ids - body_ids)
        if new:
            errors.append(f"[{section}] cites evidence that the body never uses (no new claims in abstract/conclusions): {', '.join(new)}")

    print("data:", actual)
    print(f"checked {review}: {len(cited)} IDs cited, {len(starred)} marked ★, "
          f"ledger {'present' if ledger is not None else 'missing'}")
    for e in errors:
        print("ERROR", e)
    for w in warns:
        print("WARN ", w)
    print(f"{len(errors)} error(s), {len(warns)} warning(s)")
    sys.exit(1 if errors else 0)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["init", "search", "curate", "batches", "select", "download", "extract",
                                      "biblio", "check"])
    ap.add_argument("--project", required=True, help="project directory containing topic.json")
    ap.add_argument("--sources", nargs="*", choices=list(SEARCHERS), help="search: subset of sources")
    ap.add_argument("--ids", nargs="*", help="download/extract: only these paper IDs")
    ap.add_argument("--allow-untiered", action="store_true", help="select: treat unscreened papers as tier C")
    ap.add_argument("--review", help="check: path of the review markdown (default: newest .md in 06_review)")
    args = ap.parse_args()
    {"init": stage_init, "search": stage_search, "curate": stage_curate, "batches": stage_batches,
     "select": stage_select, "download": stage_download, "extract": stage_extract, "biblio": stage_biblio,
     "check": stage_check}[args.stage](args)


if __name__ == "__main__":
    main()
