#!/usr/bin/env python3
"""Build and query a JSON index for markdown-based AI research docs.

Source of truth: markdown files under docs/research/
Generated artifact: docs/research/index.json
No external dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_ROOT = Path("docs/research")
DEFAULT_INDEX = DEFAULT_ROOT / "index.json"
RESULTS_DIR = "results"
DOC_GLOB = "**/*.md"

HEADING_RE = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$")
LIST_ITEM_RE = re.compile(
    r"^\s*-\s+(?:\[(?P<checked>[ xX\-])\]\s+)?(?P<title>.*?)(?:\s+<!--\s*(?P<meta>.*?)\s*-->)?\s*$"
)
KV_RE = re.compile(r"(?P<key>[a-zA-Z0-9_\-]+)\s*:\s*(?P<value>[^;]+)")

ALLOWED_DOC_TYPES = {
    "roadmap",
    "study",
    "eval_report",
    "comparison",
    "claim_map",
    "paper",
}
ALLOWED_ITEM_TYPES = {"hypothesis", "experiment", "claim", "decision"}
ALLOWED_LIFECYCLE_STATUS = {
    "proposed",
    "planned",
    "active",
    "running", # allowed alias for active
    "paused",
    "completed",
    "abandoned",
    "failed",
}
ALLOWED_RESEARCH_STAGE = {
    "framing",
    "designing",
    "executing",
    "evaluating",
    "synthesizing",
    "closed",
    "portfolio", # added for roadmap
}
ALLOWED_CLAIM_STATUS = {
    "drafting",
    "pending_evidence",
    "validated",
    "refuted",
    "tainted",
}

# Legal FSM transitions for research_stage (forward + allowed backward loops)
VALID_STAGE_TRANSITIONS: dict[str, set[str]] = {
    "framing": {"designing"},
    "designing": {"executing"},
    "executing": {"evaluating"},
    "evaluating": {"synthesizing", "designing"},  # backward loop allowed
    "synthesizing": {"closed"},
    "closed": set(),  # terminal
    "portfolio": set(),  # roadmap-level, no transitions
}


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:len(value)-1].strip()
        if not inner:
            return []
        return [part.strip().strip('"\'') for part in inner.split(",")]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return value.strip('"\'')


def parse_frontmatter(text: str):
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, 0
    meta = {}
    end_line = 0
    for idx in range(1, len(lines)):
        raw = lines[idx].rstrip()
        if raw.strip() == "---":
            end_line = idx + 1
            break
        if not raw or raw.lstrip().startswith("#") or ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        meta[key.strip()] = parse_scalar(value)
    return meta, end_line


def parse_inline_meta(meta_text: str):
    if not meta_text:
        return {}
    result = {}
    for match in KV_RE.finditer(meta_text):
        result[match.group("key")] = parse_scalar(match.group("value"))
    return result


def scan_root(root: Path, index_path: Path) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen_ids: dict[str, str] = {}

    for file_path in sorted(root.glob(DOC_GLOB)):
        if file_path.resolve() == index_path.resolve():
            continue

        text = file_path.read_text(encoding="utf-8")
        rel = file_path.relative_to(root).as_posix()
        meta, frontmatter_end = parse_frontmatter(text)
        lines = text.splitlines()

        if not meta:
            errors.append({"file": rel, "message": "missing frontmatter"})
            continue

        doc_id = meta.get("id", "")
        doc_type = meta.get("doc_type", "")
        lifecycle_status = str(meta.get("lifecycle_status", "")).lower()
        research_stage = str(meta.get("research_stage", "")).lower()
        previous_stage = str(meta.get("previous_stage", "")).lower()
        study_branch = meta.get("study_branch", "")

        if not doc_id:
            errors.append({"file": rel, "message": "missing id in frontmatter"})
        elif doc_id in seen_ids:
            errors.append({"file": rel, "message": f"duplicate id: {doc_id}", "other_file": seen_ids[doc_id]})
        else:
            seen_ids[doc_id] = rel

        if doc_type and doc_type not in ALLOWED_DOC_TYPES:
            errors.append({"file": rel, "message": f"invalid doc_type: {doc_type}"})
        if lifecycle_status and lifecycle_status not in ALLOWED_LIFECYCLE_STATUS:
            errors.append({"file": rel, "message": f"invalid lifecycle_status: {lifecycle_status}"})
        if research_stage and research_stage not in ALLOWED_RESEARCH_STAGE:
            errors.append({"file": rel, "message": f"invalid research_stage: {research_stage}"})

        # FSM transition validation: check previous_stage → research_stage is legal
        if previous_stage and research_stage and previous_stage in VALID_STAGE_TRANSITIONS:
            allowed_next = VALID_STAGE_TRANSITIONS[previous_stage]
            if research_stage not in allowed_next:
                errors.append({
                    "file": rel,
                    "message": f"illegal FSM transition: {previous_stage} -> {research_stage}",
                    "allowed": sorted(allowed_next),
                })

        headings = []
        doc_items = []
        for line_no, line in enumerate(lines, start=1):
            heading_match = HEADING_RE.match(line)
            if heading_match:
                headings.append(
                    {
                        "title": heading_match.group("title"),
                        "level": len(heading_match.group("level")),
                        "line": line_no,
                    }
                )

            item_match = LIST_ITEM_RE.match(line)
            if not item_match:
                continue

            item_meta = parse_inline_meta(item_match.group("meta"))
            item_id = item_meta.get("id", "")
            if not item_id:
                continue

            item_type = item_meta.get("item_type", "")
            item_lifecycle_status = str(item_meta.get("lifecycle_status", "")).lower()
            item_claim_status = str(item_meta.get("claim_status", "")).lower()
            code_branch = item_meta.get("code_branch", "")

            # Infer defaults based on checkbox
            chk = (item_match.group("checked") or "").lower()
            if not item_lifecycle_status and not item_claim_status:
                if chk == "x":
                    item_lifecycle_status = "completed"
                elif chk == "-":
                    item_lifecycle_status = "abandoned"
                elif chk == " ":
                    item_lifecycle_status = "planned"

            if item_id in seen_ids:
                errors.append({"file": rel, "message": f"duplicate id: {item_id}", "other_file": seen_ids[item_id]})
            else:
                seen_ids[item_id] = rel

            if item_type not in ALLOWED_ITEM_TYPES:
                errors.append({"file": rel, "message": f"invalid item_type: {item_type}", "id": item_id})
            
            if item_type == "claim":
                if item_claim_status and item_claim_status not in ALLOWED_CLAIM_STATUS:
                    errors.append({"file": rel, "message": f"invalid claim_status: {item_claim_status}", "id": item_id})
            else:
                if item_lifecycle_status and item_lifecycle_status not in ALLOWED_LIFECYCLE_STATUS:
                    errors.append({"file": rel, "message": f"invalid lifecycle_status: {item_lifecycle_status}", "id": item_id})

            evidence = item_meta.get("evidence_ids", [])
            if isinstance(evidence, str):
                 evidence = [e.strip() for e in evidence.split(",") if e.strip()]

            record = {
                "id": item_id,
                "item_type": item_type,
                "title": item_match.group("title").strip(),
                "lifecycle_status": item_lifecycle_status,
                "claim_status": item_claim_status,
                "research_stage": research_stage,
                "parent_id": item_meta.get("parent_id", doc_id),
                "required_gate": item_meta.get("required_gate", ""),
                "gate_status": item_meta.get("gate_status", ""),
                "evidence_ids": evidence,
                "code_branch": code_branch,
                "file": rel,
                "line": line_no,
            }
            # Cleanup clear empties to keep index tight
            record = {k: v for k, v in record.items() if v}
            
            doc_items.append(record)
            items.append(record)

        title = meta.get("title")
        if not title:
            first_heading = next((entry for entry in headings if entry["level"] == 1), None)
            title = first_heading["title"] if first_heading else file_path.stem

        documents.append(
            {
                "id": doc_id,
                "doc_type": doc_type,
                "title": title,
                "lifecycle_status": lifecycle_status,
                "research_stage": research_stage,
                "priority": meta.get("priority", ""),
                "parent_id": meta.get("parent_id", ""),
                "roadmap_id": meta.get("roadmap_id", ""),
                "study_branch": study_branch,
                "file": rel,
                "frontmatter_line_end": frontmatter_end,
                "headings": headings,
                "item_count": len(doc_items),
                "open_experiments": sum(
                    1
                    for item in doc_items
                    if item.get("item_type") == "experiment" and item.get("lifecycle_status") not in {"completed", "abandoned", "failed"}
                ),
            }
        )

    # --- Post-scan validation ---

    # Build a lookup of all known IDs -> record for referential integrity
    all_records_by_id: dict[str, dict[str, Any]] = {}
    for doc in documents:
        if doc.get("id"):
            all_records_by_id[doc["id"]] = doc
    for item in items:
        if item.get("id"):
            all_records_by_id[item["id"]] = item

    # 1. Referential integrity: parent_id must exist
    for doc in documents:
        pid = doc.get("parent_id", "")
        if pid and pid not in all_records_by_id:
            errors.append({"file": doc.get("file", ""), "message": f"broken parent_id reference: {pid}", "id": doc.get("id", "")})
    for item in items:
        pid = item.get("parent_id", "")
        if pid and pid not in all_records_by_id:
            errors.append({"file": item.get("file", ""), "message": f"broken parent_id reference: {pid}", "id": item.get("id", "")})

    # 2. Referential integrity: evidence_ids must exist
    for item in items:
        for eid in item.get("evidence_ids", []):
            if eid not in all_records_by_id:
                errors.append({"file": item.get("file", ""), "message": f"broken evidence_id reference: {eid}", "id": item.get("id", "")})

    # 3. Soft warnings: completed experiments should have results/ folder with key outputs
    warnings: list[dict[str, Any]] = []
    results_root = root / RESULTS_DIR
    for item in items:
        if item.get("item_type") != "experiment":
            continue
        if item.get("lifecycle_status") not in {"completed"}:
            continue
        exp_id = item.get("id", "")
        exp_results_dir = results_root / exp_id
        if not exp_results_dir.exists():
            warnings.append({
                "file": item.get("file", ""),
                "id": exp_id,
                "message": f"completed experiment {exp_id} has no results directory at {RESULTS_DIR}/{exp_id}/",
                "severity": "warning",
            })
        else:
            metrics_path = exp_results_dir / "metrics.json"
            config_path = exp_results_dir / "config.yaml"
            if not metrics_path.exists():
                warnings.append({
                    "file": item.get("file", ""),
                    "id": exp_id,
                    "message": f"{RESULTS_DIR}/{exp_id}/metrics.json not found",
                    "severity": "warning",
                })
            if not config_path.exists():
                warnings.append({
                    "file": item.get("file", ""),
                    "id": exp_id,
                    "message": f"{RESULTS_DIR}/{exp_id}/config.yaml not found",
                    "severity": "warning",
                })

    # 4. Taint propagation: claims referencing abandoned/failed experiments must be tainted/refuted
    for item in items:
        if item.get("item_type") != "claim":
            continue
        claim_status = item.get("claim_status", "")
        for eid in item.get("evidence_ids", []):
            evidence_record = all_records_by_id.get(eid)
            if not evidence_record:
                continue  # already caught by referential integrity check above
            ev_lifecycle = evidence_record.get("lifecycle_status", "")
            if ev_lifecycle in {"abandoned", "failed"} and claim_status not in {"tainted", "refuted"}:
                errors.append({
                    "file": item.get("file", ""),
                    "message": f"taint violation: claim {item.get('id', '')} references {ev_lifecycle} experiment {eid} but claim_status is '{claim_status}' (must be 'tainted' or 'refuted')",
                    "id": item.get("id", ""),
                })

    return {
        "project": {
            "name": next((doc["title"] for doc in documents if doc["doc_type"] == "roadmap"), ""),
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "root": root.as_posix(),
        },
        "documents": documents,
        "items": items,
        "errors": errors,
        "warnings": warnings,
    }


def write_index(data, index_path: Path):
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(index_path: Path) -> dict[str, Any]:
    if not index_path.exists():
        raise FileNotFoundError(f"index file not found: {index_path.as_posix()}")
    return json.loads(index_path.read_text(encoding="utf-8"))


def filter_records(records, args):
    text_filter = (args.text or "").lower()
    results = []
    for record in records:
        if args.doc_type and str(record.get("doc_type", "")) != args.doc_type:
            continue
        if args.item_type and str(record.get("item_type", "")) != args.item_type:
            continue
        if args.lifecycle_status and str(record.get("lifecycle_status", "")) != args.lifecycle_status:
            continue
        if args.research_stage and str(record.get("research_stage", "")) != args.research_stage:
            continue
        if args.parent_id and str(record.get("parent_id", "")) != args.parent_id:
            continue
        if text_filter:
            haystack = " ".join(
                str(record.get(field, ""))
                for field in ("id", "title", "doc_type", "item_type", "lifecycle_status", "research_stage", "parent_id")
            ).lower()
            if text_filter not in haystack:
                continue
        results.append(record)
    return results


def cmd_build(args):
    root = Path(args.root)
    index_path = root / "index.json"
    data = scan_root(root, index_path)
    write_index(data, index_path)
    summary = {
        "documents": len(data["documents"]),
        "items": len(data["items"]),
        "errors": len(data["errors"]),
        "warnings": len(data["warnings"]),
        "index": index_path.as_posix(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if data["warnings"]:
        print("\nWarnings:", file=sys.stderr)
        for w in data["warnings"]:
            print(f"  [{w['severity']}] {w['message']}", file=sys.stderr)
    return 0 if not data["errors"] else 1


def cmd_query(args):
    root = Path(args.root)
    index_path = root / "index.json"
    data = load_index(index_path)
    records = data["items"] if args.scope == "items" else data["documents"]
    print(json.dumps(filter_records(records, args), ensure_ascii=False, indent=2))
    return 0


def cmd_locate(args):
    root = Path(args.root)
    index_path = root / "index.json"
    data = load_index(index_path)
    for collection_name in ("documents", "items"):
        for record in data[collection_name]:
            if record.get("id") == args.id:
                print(json.dumps(record, ensure_ascii=False, indent=2))
                return 0
    print(json.dumps({"error": f"id not found: {args.id}"}, ensure_ascii=False, indent=2))
    return 1


def cmd_validate(args):
    root = Path(args.root)
    index_path = root / "index.json"
    data = scan_root(root, index_path)
    print(json.dumps(data["errors"], ensure_ascii=False, indent=2))
    if data["warnings"]:
        print("\nWarnings:", file=sys.stderr)
        for w in data["warnings"]:
            print(f"  [{w['severity']}] {w['message']}", file=sys.stderr)
    return 0 if not data["errors"] else 1


def cmd_show(args):
    root = Path(args.root)
    index_path = root / "index.json"
    data = load_index(index_path) if index_path.exists() else scan_root(root, index_path)

    docs_by_status: dict[str, int] = {}
    items_by_type: dict[str, int] = {}
    for obj in data["documents"]:
        doc: dict[str, Any] = obj
        docs_by_status[doc.get("lifecycle_status", "unknown")] = docs_by_status.get(doc.get("lifecycle_status", "unknown"), 0) + 1
    for obj in data["items"]:
        item: dict[str, Any] = obj
        items_by_type[item.get("item_type", "unknown")] = items_by_type.get(item.get("item_type", "unknown"), 0) + 1

    print(
        json.dumps(
            {
                "documents": len(data["documents"]),
                "items": len(data["items"]),
                "errors": len(data["errors"]),
                "documents_by_status": docs_by_status,
                "items_by_type": items_by_type,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_scaffold(args):
    """Create the standard docs/research/ directory structure."""
    root = Path(args.root)
    dirs = [
        root / "studies",
        root / "reports",
        root / "claim-maps",
        root / "comparisons",
        root / "drafts",
        root / RESULTS_DIR,
    ]
    created = []
    for d in dirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(d.relative_to(root).as_posix())
    print(json.dumps({
        "root": root.as_posix(),
        "created": created,
        "message": "scaffold complete" if created else "all directories already exist",
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_snapshot(args):
    """Copy key outputs from an experiment branch into results/<exp-id>/.

    Scans the current working directory for canonical output files
    (metrics.json, config.yaml, train_log.csv, figures/) and copies them
    into the results directory. Generates manifest.yaml automatically.
    """
    import hashlib
    import shutil

    root = Path(args.root)
    exp_id = args.exp_id
    source = Path(args.source) if args.source else Path.cwd()
    target = root / RESULTS_DIR / exp_id
    target.mkdir(parents=True, exist_ok=True)

    # Files to look for and copy
    canonical_files = ["metrics.json", "config.yaml", "train_log.csv", "README.md"]
    canonical_dirs = ["figures"]

    copied = []
    manifest_entries = []

    for fname in canonical_files:
        src = source / fname
        if src.exists():
            shutil.copy2(src, target / fname)
            copied.append(fname)
            file_hash = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
            manifest_entries.append({
                "file": fname,
                "size_bytes": src.stat().st_size,
                "sha256_prefix": file_hash,
            })

    for dname in canonical_dirs:
        src = source / dname
        if src.exists() and src.is_dir():
            dst = target / dname
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            file_count = sum(1 for _ in dst.rglob("*") if _.is_file())
            copied.append(f"{dname}/ ({file_count} files)")
            manifest_entries.append({
                "file": f"{dname}/",
                "file_count": file_count,
            })

    # Also copy any extra files the user specified
    for extra in (args.extra or []):
        src = source / extra
        if src.exists():
            dst = target / extra
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            copied.append(extra)
            if src.is_file():
                file_hash = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
                manifest_entries.append({
                    "file": extra,
                    "size_bytes": src.stat().st_size,
                    "sha256_prefix": file_hash,
                })

    # Write manifest.yaml (auto-generated)
    manifest_path = target / "manifest.yaml"
    manifest_lines = [
        f"# Auto-generated by research_index.py snapshot",
        f"# {datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')}",
        f"experiment_id: {exp_id}",
        f"source: {source.as_posix()}",
        f"files:",
    ]
    for entry in manifest_entries:
        manifest_lines.append(f"  - file: {entry['file']}")
        for k, v in entry.items():
            if k != "file":
                manifest_lines.append(f"    {k}: {v}")
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "experiment_id": exp_id,
        "target": target.as_posix(),
        "copied": copied,
        "manifest": manifest_path.as_posix(),
    }, ensure_ascii=False, indent=2))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Index and query markdown-based AI research docs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--root", default=DEFAULT_ROOT.as_posix())
    build_parser.set_defaults(func=cmd_build)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("--root", default=DEFAULT_ROOT.as_posix())
    query_parser.add_argument("--scope", choices=["documents", "items"], default="items")
    query_parser.add_argument("--doc_type")
    query_parser.add_argument("--item_type")
    query_parser.add_argument("--lifecycle_status")
    query_parser.add_argument("--research_stage")
    query_parser.add_argument("--parent_id")
    query_parser.add_argument("--text")
    query_parser.set_defaults(func=cmd_query)

    locate_parser = subparsers.add_parser("locate")
    locate_parser.add_argument("--root", default=DEFAULT_ROOT.as_posix())
    locate_parser.add_argument("--id", required=True)
    locate_parser.set_defaults(func=cmd_locate)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--root", default=DEFAULT_ROOT.as_posix())
    validate_parser.set_defaults(func=cmd_validate)

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("--root", default=DEFAULT_ROOT.as_posix())
    show_parser.set_defaults(func=cmd_show)

    scaffold_parser = subparsers.add_parser("scaffold", help="Create standard directory structure")
    scaffold_parser.add_argument("--root", default=DEFAULT_ROOT.as_posix())
    scaffold_parser.set_defaults(func=cmd_scaffold)

    snapshot_parser = subparsers.add_parser("snapshot", help="Copy experiment outputs into results/")
    snapshot_parser.add_argument("--root", default=DEFAULT_ROOT.as_posix())
    snapshot_parser.add_argument("--exp-id", required=True, help="Experiment ID (e.g., exp-001)")
    snapshot_parser.add_argument("--source", default=None, help="Source directory (default: cwd)")
    snapshot_parser.add_argument("--extra", nargs="*", help="Extra files/dirs to copy")
    snapshot_parser.set_defaults(func=cmd_snapshot)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())