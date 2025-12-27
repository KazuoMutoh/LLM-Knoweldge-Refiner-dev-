import random
from pathlib import Path

import pytest

from simple_active_refine.amie import AmieRule, TriplePattern
from simple_active_refine.pipeline import RefinedKG, TripleAcquisitionContext
from simple_active_refine.triple_acquirer_impl import (
    RuleBasedTripleAcquirer,
    RandomTripleAcquirer,
)


def _write_tsv(path: Path, triples):
    path.write_text("\n".join(["\t".join(t) for t in triples]) + "\n", encoding="utf-8")


def _make_rule(head_rel: str, body_rel1: str, body_rel2: str) -> AmieRule:
    head = TriplePattern("?x", head_rel, "?y")
    body = [
        TriplePattern("?x", body_rel1, "?z"),
        TriplePattern("?z", body_rel2, "?y"),
    ]
    return AmieRule(head=head, body=body, support=None, std_conf=None, pca_conf=None,
                    head_coverage=None, body_size=None, pca_body_size=None, raw=f"{body_rel1}+{body_rel2}")


def test_rule_based_acquirer_uses_train_removed(tmp_path: Path):
    # Setup minimal dataset
    train = [("alice", "/knows", "bob")]
    removed = [
        ("alice", "/born_in", "kyoto"),
        ("kyoto", "/located_in", "japan"),
        ("paris", "/located_in", "france"),  # noise, not connected
    ]
    _write_tsv(tmp_path / "train.txt", train)
    _write_tsv(tmp_path / "train_removed.txt", removed)

    rule = _make_rule(head_rel="/people/person/nationality", body_rel1="/born_in", body_rel2="/located_in")
    target_triples = [("alice", "/people/person/nationality", "japan")]

    acquirer = RuleBasedTripleAcquirer(
        target_triples=target_triples,
        candidate_dir=str(tmp_path),
        n_targets_per_rule=1,
    )

    ctx = TripleAcquisitionContext(kg=RefinedKG(triples=train), rules=[rule], iteration=1)
    result = acquirer.acquire(ctx)

    key = str(rule)
    assert key in result.candidates_by_rule
    added = set(result.candidates_by_rule[key])
    assert ("alice", "/born_in", "kyoto") in added
    assert ("kyoto", "/located_in", "japan") in added
    assert result.diagnostics["n_total_candidates"] == len(added)


def test_random_acquirer_samples_from_train_removed(tmp_path: Path):
    train = [("h0", "r0", "t0")]
    removed = [
        ("h1", "r1", "t1"),
        ("h2", "r2", "t2"),
        ("noise_h", "noise_r", "noise_t"),
    ]
    _write_tsv(tmp_path / "train.txt", train)
    _write_tsv(tmp_path / "train_removed.txt", removed)

    rule = _make_rule(head_rel="/p", body_rel1="/q1", body_rel2="/q2")
    random.seed(0)
    acquirer = RandomTripleAcquirer(
        candidate_dir=str(tmp_path),
        n_triples_per_rule=1,
    )
    ctx = TripleAcquisitionContext(kg=RefinedKG(triples=train), rules=[rule], iteration=1)
    result = acquirer.acquire(ctx)

    key = str(rule)
    assert key in result.candidates_by_rule
    picked = result.candidates_by_rule[key]
    assert len(picked) == 1
    assert picked[0] in removed
    assert result.diagnostics["n_total_candidates"] == 1
