#!/usr/bin/env python3
"""Test suite for research_index.py validation fixes.

Tests taint propagation, FSM transition validation, referential integrity,
and item_type semantic correctness.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Add the scripts directory to path so we can import research_index
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import research_index as ri


def write_md(root: Path, relpath: str, content: str):
    """Helper to write a markdown file under root."""
    fp = root / relpath
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content, encoding="utf-8")


def find_errors(errors: list, substring: str) -> list:
    """Find errors whose message contains the given substring."""
    return [e for e in errors if substring in e.get("message", "")]


def test_valid_fixture(root: Path, index_path: Path):
    """A fully valid fixture should produce zero errors."""
    write_md(root, "RESEARCH_ROADMAP.md", """---
id: research-roadmap-main
doc_type: roadmap
lifecycle_status: active
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Test Roadmap
""")
    write_md(root, "studies/study-001.md", """---
id: rs-001
doc_type: study
title: Test Study
lifecycle_status: active
research_stage: framing
previous_stage: ""
roadmap_id: research-roadmap-main
study_branch: study/rs-001
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Test Study

## Hypotheses

- H1: Test hypothesis <!-- id: hyp-001; item_type: hypothesis; lifecycle_status: proposed -->

## Experiments

- [ ] Baseline <!-- id: exp-001; item_type: experiment; parent_id: hyp-001; code_branch: exp/exp-001; required_gate: G2_approval; lifecycle_status: planned -->
- [x] Treatment <!-- id: exp-002; item_type: experiment; parent_id: hyp-001; code_branch: exp/exp-002; lifecycle_status: completed -->

## Claims

- Claim: Main claim <!-- id: clm-001; item_type: claim; evidence_ids: exp-002; claim_status: pending_evidence -->
""")
    write_md(root, "reports/eval-001.md", """---
id: eval-001
doc_type: eval_report
title: Eval for EXP-002
lifecycle_status: completed
parent_id: rs-001
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Eval
""")

    data = ri.scan_root(root, index_path)
    errors = data["errors"]
    assert len(errors) == 0, f"Expected 0 errors for valid fixture, got {len(errors)}: {json.dumps(errors, indent=2)}"
    print("  [PASS] Valid fixture produces 0 errors")

    # Check item_type is used (not doc_type) for inline items
    items = data["items"]
    for item in items:
        assert "item_type" in item, f"Expected 'item_type' key in item record, got keys: {list(item.keys())}"
        assert "doc_type" not in item, f"'doc_type' should NOT be in item record, but it is: {item}"
    print("  [PASS] item_type semantic fix verified (no doc_type in inline items)")


def test_taint_propagation(root: Path, index_path: Path):
    """Claims referencing abandoned experiments must be tainted."""
    write_md(root, "RESEARCH_ROADMAP.md", """---
id: research-roadmap-main
doc_type: roadmap
lifecycle_status: active
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Roadmap
""")
    write_md(root, "studies/taint-study.md", """---
id: rs-taint
doc_type: study
title: Taint Test
lifecycle_status: active
research_stage: evaluating
previous_stage: executing
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Taint Test

- H1: Hypothesis <!-- id: hyp-t1; item_type: hypothesis; lifecycle_status: active -->
- [ ] Abandoned exp <!-- id: exp-t1; item_type: experiment; parent_id: hyp-t1; lifecycle_status: abandoned -->
- [x] Good exp <!-- id: exp-t2; item_type: experiment; parent_id: hyp-t1; lifecycle_status: completed -->
- Claim: Bad claim still drafting <!-- id: clm-t1; item_type: claim; evidence_ids: exp-t1, exp-t2; claim_status: drafting -->
- Claim: Correctly tainted claim <!-- id: clm-t2; item_type: claim; evidence_ids: exp-t1; claim_status: tainted -->
""")

    data = ri.scan_root(root, index_path)
    errors = data["errors"]

    taint_errors = find_errors(errors, "taint violation")
    assert len(taint_errors) >= 1, f"Expected at least 1 taint violation, got {len(taint_errors)}"
    assert any("clm-t1" in e["message"] for e in taint_errors), "clm-t1 should be flagged for taint violation"
    assert not any("clm-t2" in e["message"] for e in taint_errors), "clm-t2 should NOT be flagged (already tainted)"
    print("  [PASS] Taint propagation correctly detects untainted claims with abandoned evidence")


def test_fsm_illegal_transition(root: Path, index_path: Path):
    """Illegal FSM transitions (e.g., framing -> executing) must be caught."""
    write_md(root, "RESEARCH_ROADMAP.md", """---
id: research-roadmap-main
doc_type: roadmap
lifecycle_status: active
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Roadmap
""")
    write_md(root, "studies/fsm-bad.md", """---
id: rs-fsm-bad
doc_type: study
title: Bad FSM
lifecycle_status: active
research_stage: executing
previous_stage: framing
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Bad FSM - skipped designing stage!
""")

    data = ri.scan_root(root, index_path)
    errors = data["errors"]

    fsm_errors = find_errors(errors, "illegal FSM transition")
    assert len(fsm_errors) >= 1, f"Expected FSM error for framing->executing, got {len(fsm_errors)}"
    assert any("framing -> executing" in e["message"] for e in fsm_errors), \
        f"Expected 'framing -> executing' in FSM error message, got: {fsm_errors}"
    print("  [PASS] Illegal FSM transition (framing -> executing) correctly detected")


def test_fsm_legal_backward_loop(root: Path, index_path: Path):
    """evaluating -> designing is a legal backward loop."""
    write_md(root, "RESEARCH_ROADMAP.md", """---
id: research-roadmap-main
doc_type: roadmap
lifecycle_status: active
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Roadmap
""")
    write_md(root, "studies/fsm-good.md", """---
id: rs-fsm-good
doc_type: study
title: Good FSM Backward Loop
lifecycle_status: active
research_stage: designing
previous_stage: evaluating
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Good FSM - legal backward loop
""")

    data = ri.scan_root(root, index_path)
    errors = data["errors"]

    fsm_errors = find_errors(errors, "illegal FSM transition")
    assert len(fsm_errors) == 0, f"evaluating->designing should be legal, got errors: {fsm_errors}"
    print("  [PASS] Legal backward loop (evaluating -> designing) accepted")


def test_broken_parent_id(root: Path, index_path: Path):
    """parent_id pointing to nonexistent ID must be caught."""
    write_md(root, "RESEARCH_ROADMAP.md", """---
id: research-roadmap-main
doc_type: roadmap
lifecycle_status: active
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Roadmap
""")
    write_md(root, "reports/broken-parent.md", """---
id: eval-broken
doc_type: eval_report
title: Broken parent ref
lifecycle_status: completed
parent_id: rs-nonexistent
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Eval with broken parent
""")

    data = ri.scan_root(root, index_path)
    errors = data["errors"]

    ref_errors = find_errors(errors, "broken parent_id reference")
    assert len(ref_errors) >= 1, f"Expected broken parent_id error, got {len(ref_errors)}"
    assert any("rs-nonexistent" in e["message"] for e in ref_errors), \
        f"Expected 'rs-nonexistent' in error message, got: {ref_errors}"
    print("  [PASS] Broken parent_id reference correctly detected")


def test_broken_evidence_ids(root: Path, index_path: Path):
    """evidence_ids pointing to nonexistent experiment must be caught."""
    write_md(root, "RESEARCH_ROADMAP.md", """---
id: research-roadmap-main
doc_type: roadmap
lifecycle_status: active
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Roadmap
""")
    write_md(root, "studies/broken-evidence.md", """---
id: rs-bev
doc_type: study
title: Broken evidence
lifecycle_status: active
research_stage: evaluating
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Broken evidence

- Claim: Ghost evidence <!-- id: clm-bev; item_type: claim; evidence_ids: exp-999; claim_status: pending_evidence -->
""")

    data = ri.scan_root(root, index_path)
    errors = data["errors"]

    ref_errors = find_errors(errors, "broken evidence_id reference")
    assert len(ref_errors) >= 1, f"Expected broken evidence_id error, got {len(ref_errors)}"
    assert any("exp-999" in e["message"] for e in ref_errors), \
        f"Expected 'exp-999' in error message, got: {ref_errors}"
    print("  [PASS] Broken evidence_id reference correctly detected")


def test_results_warnings(root: Path, index_path: Path):
    """Completed experiments without results/ folder should produce warnings."""
    write_md(root, "RESEARCH_ROADMAP.md", """---
id: research-roadmap-main
doc_type: roadmap
lifecycle_status: active
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Roadmap
""")
    write_md(root, "studies/warn-study.md", """---
id: rs-warn
doc_type: study
title: Warning Test
lifecycle_status: active
research_stage: evaluating
previous_stage: executing
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Warning Test

- H1: Hypothesis <!-- id: hyp-w1; item_type: hypothesis; lifecycle_status: active -->
- [x] Completed without results <!-- id: exp-w1; item_type: experiment; parent_id: hyp-w1; lifecycle_status: completed -->
- [ ] Still planned <!-- id: exp-w2; item_type: experiment; parent_id: hyp-w1; lifecycle_status: planned -->
""")

    data = ri.scan_root(root, index_path)
    errors = data["errors"]
    warnings = data["warnings"]

    assert len(errors) == 0, f"Expected 0 errors, got {len(errors)}: {json.dumps(errors, indent=2)}"
    assert len(warnings) >= 1, f"Expected at least 1 warning for missing results, got {len(warnings)}"
    assert any("exp-w1" in w["message"] for w in warnings), "exp-w1 should have a missing results warning"
    assert not any("exp-w2" in w.get("message", "") for w in warnings), "exp-w2 (planned) should NOT have a warning"
    print("  [PASS] Completed experiment without results/ produces soft warning")
    print("  [PASS] Planned experiment does not produce warning")


def test_results_no_warning_when_present(root: Path, index_path: Path):
    """Completed experiments with results/ folder should NOT produce warnings."""
    write_md(root, "RESEARCH_ROADMAP.md", """---
id: research-roadmap-main
doc_type: roadmap
lifecycle_status: active
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Roadmap
""")
    write_md(root, "studies/ok-study.md", """---
id: rs-ok
doc_type: study
title: No Warning Test
lifecycle_status: active
research_stage: evaluating
previous_stage: executing
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# No Warning Test

- H1: Hypothesis <!-- id: hyp-ok1; item_type: hypothesis; lifecycle_status: active -->
- [x] Has results <!-- id: exp-ok1; item_type: experiment; parent_id: hyp-ok1; lifecycle_status: completed -->
""")

    # Create results directory with key files
    results_dir = root / "results" / "exp-ok1"
    results_dir.mkdir(parents=True)
    (results_dir / "metrics.json").write_text('{"accuracy": 0.95}', encoding="utf-8")
    (results_dir / "config.yaml").write_text('learning_rate: 0.001', encoding="utf-8")

    data = ri.scan_root(root, index_path)
    warnings = data["warnings"]
    assert len(warnings) == 0, f"Expected 0 warnings when results exist, got {len(warnings)}: {json.dumps(warnings, indent=2)}"
    print("  [PASS] Completed experiment with results/ produces no warning")


def test_scaffold(root: Path, index_path: Path):
    """Scaffold command should create all expected directories."""
    # Run scaffold by calling cmd_scaffold with args
    class Args:
        pass
    args = Args()
    args.root = root.as_posix()
    ri.cmd_scaffold(args)

    expected = ["studies", "reports", "claim-maps", "comparisons", "drafts", "results"]
    for d in expected:
        assert (root / d).exists(), f"Expected {d}/ to exist after scaffold"
    print("  [PASS] Scaffold creates all expected directories")


def test_comparison_and_decision_contracts(root: Path, index_path: Path):
    """Comparison docs and decision items should validate as first-class contracts."""
    write_md(root, "RESEARCH_ROADMAP.md", """---
id: research-roadmap-main
doc_type: roadmap
lifecycle_status: active
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Roadmap
""")
    write_md(root, "studies/contract-study.md", """---
id: rs-contract
doc_type: study
title: Contract Test
lifecycle_status: active
research_stage: framing
previous_stage: ""
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Contract Test

- H1: Hypothesis <!-- id: hyp-contract; item_type: hypothesis; lifecycle_status: proposed -->
- Keep scope narrow <!-- id: dec-contract; item_type: decision; lifecycle_status: active -->
""")
    write_md(root, "comparisons/compare-001.md", """---
id: cmp-001
doc_type: comparison
title: Comparison Contract
lifecycle_status: active
parent_id: rs-contract
managed_by: scripts/research_index.py
source_of_truth: markdown
---
# Comparison Contract
""")

    data = ri.scan_root(root, index_path)
    errors = data["errors"]
    assert len(errors) == 0, f"Expected 0 errors for comparison/decision contract fixture, got {len(errors)}: {json.dumps(errors, indent=2)}"
    assert any(doc.get("doc_type") == "comparison" for doc in data["documents"]), "Expected comparison doc_type to be indexed"
    assert any(item.get("item_type") == "decision" for item in data["items"]), "Expected decision item_type to be indexed"
    print("  [PASS] comparison doc_type and decision item_type validate correctly")


def run_all_tests():
    """Run all tests in isolated temp directories."""
    tests = [
        ("Test 1: Valid fixture (zero errors)", test_valid_fixture),
        ("Test 2: Taint propagation", test_taint_propagation),
        ("Test 3: Illegal FSM transition", test_fsm_illegal_transition),
        ("Test 4: Legal backward loop", test_fsm_legal_backward_loop),
        ("Test 5: Broken parent_id", test_broken_parent_id),
        ("Test 6: Broken evidence_ids", test_broken_evidence_ids),
        ("Test 7: Results warnings (missing)", test_results_warnings),
        ("Test 8: Results warnings (present)", test_results_no_warning_when_present),
        ("Test 9: Scaffold command", test_scaffold),
        ("Test 10: Comparison and decision contracts", test_comparison_and_decision_contracts),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n{name}:")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                index_path = root / "index.json"
                test_fn(root, index_path)
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'='*50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
