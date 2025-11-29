from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable
from .amie import Rule
from .kg import KG, Triple


@dataclass
class RuleFilter:
    min_std_conf: float = 0.2
    min_pca_conf: float = 0.2
    min_head_cov: float = 0.0
    top_k: int = 50


def filter_rules(rules: List[Rule], rf: RuleFilter) -> List[Rule]:
    """Filter and rank AMIE rules for triple addition purposes.

    Ranking priority: PCA confidence, then std confidence, then head coverage.
    """
    cand = [r for r in rules
            if (r.pca_conf is None or r.pca_conf >= rf.min_pca_conf)
            and (r.std_conf is None or r.std_conf >= rf.min_std_conf)
            and (r.head_cov is None or r.head_cov >= rf.min_head_cov)]
    cand.sort(key=lambda r: (r.pca_conf or 0.0, r.std_conf or 0.0, r.head_cov or 0.0), reverse=True)
    return cand[: rf.top_k]


def missing_body_atoms_for_triple(rule: Rule, h: str, t: str, kg: KG) -> List[Triple]:
    """Given a rule Body => (?H, r_t, ?T) and a concrete (h, r_t, t),
    find body atoms missing in KG under the canonical mapping μ(?H)=h, μ(?T)=t.

    This simplified matcher assumes variables appearing only in head and once per
    body atom (AMIE‑style). It resolves intermediate variables via existence in KG
    when possible; missing atoms are returned with concrete entities if resolvable,
    or variables kept otherwise.
    """
    # Identify head variables
    Hvar, r_head, Tvar = rule.head
    assert r_head, "Head relation must be set"
    var_bind: Dict[str, str] = {Hvar: h, Tvar: t}

    missing: List[Triple] = []

    # Try a single pass: if an atom has two variables and one is bound, try to
    # resolve the other via KG neighborhood; otherwise mark as missing with vars.
    for a, r, b in rule.body:
        a_val = var_bind.get(a, a)
        b_val = var_bind.get(b, b)
        if a_val.startswith("?") and b_val.startswith("?"):
            # cannot ground; skip but mark symbolic missing
            missing.append((a_val, r, b_val))
            continue
        if not a_val.startswith("?") and not b_val.startswith("?"):
            # Fully grounded
            if not kg.has_edge(a_val, r, b_val):
                missing.append((a_val, r, b_val))
            continue
        # One side grounded, attempt to infer the other through existing edges
        if not a_val.startswith("?") and b_val.startswith("?"):
            # need a_val --r--> x such that x connects further if needed
            tails = kg.tails(a_val, r)
            if tails:
                pick = next(iter(tails))
                var_bind[b] = pick
                # Not missing because it exists
            else:
                missing.append((a_val, r, b_val))
            continue
        if a_val.startswith("?") and not b_val.startswith("?"):
            heads = kg.heads(r, b_val)
            if heads:
                pick = next(iter(heads))
                var_bind[a] = pick
            else:
                missing.append((a_val, r, b_val))
            continue
    # Replace any remaining bound vars
    concretized: List[Triple] = []
    for x in missing:
        a, r, b = x
        a = var_bind.get(a, a)
        b = var_bind.get(b, b)
        concretized.append((a, r, b))
    return concretized