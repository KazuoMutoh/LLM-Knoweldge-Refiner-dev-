"""Arm selection strategies.

This module mirrors the design of :mod:`simple_active_refine.rule_selector` but
selects *arms* (combinations of rules) instead of individual rules.

Design principles:
- Google Style Docstring
- PEP8 compliance
- Explicit type hints
- Use util.get_logger() for consistent logging

Note:
    The LLM-based selector is implemented so it can be enabled in experiments,
    but unit tests should mock the LLM to avoid external API calls.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from settings import OPENAI_API_KEY
from simple_active_refine.arm import Arm, ArmWithId
from simple_active_refine.arm_history import ArmHistory, ArmStatistics
from simple_active_refine.util import get_logger

logger = get_logger("arm_selector")

os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)


@dataclass
class ArmCandidate:
    """Arm candidate with a stable ID."""

    arm_id: str
    arm: Arm

    @classmethod
    def from_arm_with_id(cls, arm_with_id: ArmWithId) -> "ArmCandidate":
        return cls(arm_id=arm_with_id.arm_id, arm=arm_with_id.arm)


class ArmSelectionPolicy(BaseModel):
    """LLM structured output schema for arm selection."""

    reasoning: str = Field(..., description="Analysis and reasoning for the selection")
    policy_text: str = Field(..., description="Updated selection policy in natural language")
    selected_arm_ids: List[str] = Field(..., description="List of selected arm IDs")
    rationale_per_arm: Dict[str, str] = Field(
        default_factory=dict,
        description="Rationale per selected arm (optional; may be empty if not provided)",
    )


class ArmSelector:
    """Base class for arm selection."""

    def __init__(
        self,
        history: Optional[ArmHistory] = None,
        target_predicates: Optional[List[str]] = None,
        relation_texts: Optional[Dict[str, str]] = None,
        entity_texts: Optional[Dict[str, str]] = None,
    ) -> None:
        self.history = history or ArmHistory()
        self.target_predicates = list(target_predicates) if target_predicates else []
        self.relation_texts = dict(relation_texts) if relation_texts else {}
        self.entity_texts = dict(entity_texts) if entity_texts else {}

    def select_arms(
        self,
        arm_pool: List[ArmWithId],
        k: int = 3,
        iteration: int = 0,
    ) -> tuple[List[ArmWithId], Optional[str]]:
        """Select k arms from the pool.

        Args:
            arm_pool: Candidate arms.
            k: Number of arms to select.
            iteration: Current iteration.

        Returns:
            tuple[List[ArmWithId], Optional[str]]: Selected arms and optional policy text.
        """

        raise NotImplementedError


class LLMPolicyArmSelector(ArmSelector):
    """LLM policy-driven arm selection.

    The selector formats arm statistics and asks an LLM to choose k arm IDs.
    """

    def __init__(
        self,
        history: Optional[ArmHistory] = None,
        target_predicates: Optional[List[str]] = None,
        relation_texts: Optional[Dict[str, str]] = None,
        entity_texts: Optional[Dict[str, str]] = None,
        chat_model: str = "gpt-4o",
        temperature: float = 0.3,
        request_timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(
            history,
            target_predicates=target_predicates,
            relation_texts=relation_texts,
            entity_texts=entity_texts,
        )
        # Avoid hanging indefinitely on network/model issues.
        self.llm = ChatOpenAI(
            model_name=chat_model,
            temperature=temperature,
            request_timeout=float(request_timeout),
            max_retries=int(max_retries),
        )
        self.structured_llm = self.llm.with_structured_output(ArmSelectionPolicy)
        self.current_policy: Optional[str] = None

    @staticmethod
    def _truncate_text(text: str, max_len: int = 80) -> str:
        text = (text or "").strip().replace("\n", " ")
        if len(text) <= max_len:
            return text
        return text[: max_len - 1] + "…"

    def _entity_label(self, entity_id: str) -> str:
        label = self.entity_texts.get(entity_id, "")
        label = self._truncate_text(label, max_len=80)
        return label or entity_id

    def _relation_desc(self, predicate: str) -> str:
        desc = self.relation_texts.get(predicate, "")
        desc = self._truncate_text(desc, max_len=80)
        return desc

    def _format_arm_statistics(self, arm_pool: List[ArmWithId]) -> str:
        all_stats = self.history.get_all_arm_statistics()

        def _fmt_triple(t: Tuple[str, str, str]) -> str:
            s, p, o = t
            s_txt = self._entity_label(s)
            o_txt = self._entity_label(o)
            p_desc = self._relation_desc(p)
            if p_desc:
                return f"{s_txt} ({s}) — {p} [{p_desc}] — {o_txt} ({o})"
            return f"{s_txt} ({s}) — {p} — {o_txt} ({o})"

        def _top_targets_by_witness(record_targets: List[Tuple[str, str, str]], witness_by_target: Dict) -> List[str]:
            items = [(t, int(witness_by_target.get(t, 0))) for t in record_targets]
            items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
            lines: List[str] = []
            for t, w in items_sorted[:3]:
                lines.append(f"- {_fmt_triple(t)}  witness={w}")
            return lines

        def _sample_triples(triples: List[Tuple[str, str, str]], k: int = 3) -> List[str]:
            return [f"- {_fmt_triple(t)}" for t in triples[:k]]

        lines: List[str] = []
        lines.append("## Current Arm Pool Statistics\n")

        for i, awi in enumerate(arm_pool, 1):
            arm_id = awi.arm_id
            arm = awi.arm
            lines.append(f"### Arm {i}: {arm_id}")
            lines.append(f"**Type**: {arm.arm_type}")
            lines.append(f"**Rule keys**: {arm.rule_keys}")

            body_preds = arm.metadata.get("body_predicates") if isinstance(arm.metadata, dict) else None
            if isinstance(body_preds, list) and body_preds:
                lines.append(f"**Body predicates (derived)**: {body_preds}")
                body_texts = arm.metadata.get("body_predicate_texts") if isinstance(arm.metadata, dict) else None
                if isinstance(body_texts, dict) and body_texts:
                    # Keep it short: show up to 5 predicate descriptions.
                    shown = 0
                    lines.append("- **Body predicate descriptions (sample)**:")
                    for p in body_preds:
                        if shown >= 5:
                            break
                        desc = body_texts.get(p)
                        if desc:
                            lines.append(f"  - {p}: {desc}")
                            shown += 1

            if arm_id in all_stats:
                stats: ArmStatistics = all_stats[arm_id]
                lines.append(f"- **Trials**: {stats.total_iterations}")
                lines.append(f"- **Mean reward**: {stats.mean_reward:.6f}")
                lines.append(f"- **Std reward**: {stats.std_reward:.6f}")
                lines.append(f"- **Recent performance** (last 3): {stats.recent_performance:.6f}")
                lines.append(f"- **Total triples added**: {stats.total_triples_added}")

                # Add last-iteration qualitative context to judge explanatory value.
                recs = self.history.get_records_for_arm(arm_id)
                if recs:
                    last = recs[-1]
                    diag = dict(getattr(last, "diagnostics", {}) or {})
                    lines.append(f"- **Last iter**: {last.iteration}")
                    if "targets_total" in diag and "targets_with_witness" in diag:
                        lines.append(
                            f"- **Target coverage**: {diag.get('targets_with_witness', 0.0):.0f} / "
                            f"{diag.get('targets_total', 0.0):.0f}  (rate={diag.get('target_coverage', 0.0):.3f})"
                        )
                    if "mean_witness_per_target" in diag:
                        lines.append(f"- **Mean witness/target**: {diag.get('mean_witness_per_target', 0.0):.3f}")
                    if "evidence_new" in diag and "evidence_total" in diag:
                        lines.append(
                            f"- **Evidence acquired**: total={diag.get('evidence_total', 0.0):.0f}, "
                            f"new={diag.get('evidence_new', 0.0):.0f}, existing={diag.get('evidence_existing', 0.0):.0f}"
                        )
                    if "evidence_new_overlap_rate_with_targets" in diag:
                        lines.append(
                            f"- **New-evidence overlap w/ target entities**: "
                            f"{diag.get('evidence_new_overlap_rate_with_targets', 0.0):.3f}"
                        )

                    witness_by_target = getattr(last, "witness_by_target", {}) or {}
                    if last.target_triples:
                        lines.append("- **Top witnessed targets (last iter)**:")
                        lines.extend(_top_targets_by_witness(last.target_triples, witness_by_target))

                    if last.added_triples:
                        lines.append("- **New evidence added (sample)**:")
                        lines.extend(_sample_triples(last.added_triples, k=3))

                    evidence_all = getattr(last, "evidence_triples", []) or []
                    if evidence_all:
                        lines.append("- **Evidence acquired (sample)**:")
                        lines.extend(_sample_triples(evidence_all, k=3))
            else:
                lines.append("- **Trials**: 0 (UNTRIED)")
                lines.append("- **Status**: This arm has never been tested")
            lines.append("")

        return "\n".join(lines)

    def _create_selection_prompt(self, arm_pool: List[ArmWithId], k: int, iteration: int) -> str:
        stats_text = self._format_arm_statistics(arm_pool)

        # Target relation context helps semantic selection (e.g., nationality ~ geography).
        target_preds: List[str] = list(self.target_predicates)
        if not target_preds:
            # Fallback: infer from any recorded targets.
            for rec in reversed(self.history.records):
                for _, p, _ in rec.target_triples:
                    if p not in target_preds:
                        target_preds.append(p)
                if target_preds:
                    break

        target_desc_lines: List[str] = []
        for p in target_preds[:5]:
            desc = self.relation_texts.get(p)
            if desc:
                target_desc_lines.append(f"- {p}: {desc}")
            else:
                target_desc_lines.append(f"- {p}")

        target_context = "\n".join(target_desc_lines) if target_desc_lines else "- (unknown)"

        semantic_grounding = """## Semantic Grounding (must follow)

    You MUST prioritize semantic alignment in addition to proxy reward.

    Definitions:
    - A target triple (h, r_target, t) is "explained" by an arm when the arm acquires/adds evidence triples that connect to the target entities (h or t) and are semantically relevant to r_target.
    - Evidence that only increases witness/coverage but is semantically unrelated should be treated as weak or noisy.

    Rubric (use internally when deciding):
    - SemanticAlignmentScore (0-2): Do the arm's body predicates and evidence look meaningfully related to the target predicate?
    - EvidenceRelevanceScore (0-2): Do the evidence/added triples mention/overlap target entities and form plausible supporting context?
    - ProxyReliability (0-2): Is high reward supported by meaningful overlap/coverage rather than obvious inflation?

    Hard constraint:
    - If possible, at least one of the selected arms MUST have SemanticAlignmentScore >= 1.
      If none qualify, explicitly say so in reasoning and select the best available (fallback).
    """

        prompt = f"""You are an expert in knowledge graph refinement using reinforcement learning principles.

Your task is to select {k} arms from the arm pool for iteration {iteration}.

## Target Relation Context
The target predicate(s) (what we are trying to explain) are:
{target_context}

    {semantic_grounding}

{stats_text}

## Current Selection Policy
{self.current_policy if self.current_policy else "No policy yet (first iteration). Start with exploration."}

## Your Task

1. Analyze the statistics above carefully, including the *meaning* of each arm (rule keys) and the *actual evidence triples* it acquired/added.
2. Judge whether each arm was truly helpful to explain the target triples it evaluated.
    - High target coverage (many targets have witnesses) is good.
    - Evidence should be relevant to targets (e.g., overlap with target entities).
    - Beware arms that add many triples but have low witness/coverage: may be noisy or off-target.
        - Consider semantic alignment as a FIRST-CLASS objective: prefer arms whose body predicates and evidence are semantically relevant to the target predicate(s).
            Example: for nationality, geographic/location-related predicates are often more useful than friendship-only predicates.
        - When you cite a triple example, interpret it using the provided entity/relation text (when available).
3. Update the policy text (natural language) with concrete, reusable criteria balancing exploration vs exploitation.
4. Select {k} arm IDs from the pool.

## Guidelines
- Exploration: prioritize untried/undertried arms early.
- Exploitation: favor arms with high mean reward and stable recent performance.
- Explanatory usefulness: prefer arms whose last-iteration evidence/coverage indicates they explain targets well.
- Keep selection within the provided arm IDs.

## Output Format
Provide a JSON with:
- reasoning
- policy_text
- selected_arm_ids
- rationale_per_arm
"""

        return prompt

    def select_arms(
        self,
        arm_pool: List[ArmWithId],
        k: int = 3,
        iteration: int = 0,
    ) -> tuple[List[ArmWithId], Optional[str]]:
        if len(arm_pool) <= k:
            logger.info("Pool size (%d) <= k (%d), selecting all arms", len(arm_pool), k)
            return arm_pool, self.current_policy

        # Iteration 0: respect the existing pool order.
        if iteration == 0:
            logger.info("Iteration 0: selecting first %d arms in pool order", k)
            selected = arm_pool[:k]
            if self.current_policy is None:
                self.current_policy = (
                    "Initial iteration: Used the pre-ranked arm pool order. "
                    "Later iterations adapt based on observed performance while keeping exploration/exploitation balance."
                )
            return selected, self.current_policy

        prompt = self._create_selection_prompt(arm_pool, k, iteration)
        logger.info("Consulting LLM for arm selection (iteration %d)", iteration)

        try:
            result: ArmSelectionPolicy = self.structured_llm.invoke(prompt)
            self.current_policy = result.policy_text

            arm_id_to_arm = {a.arm_id: a for a in arm_pool}
            selected: List[ArmWithId] = []

            for arm_id in result.selected_arm_ids:
                if arm_id in arm_id_to_arm:
                    selected.append(arm_id_to_arm[arm_id])
                    rationale = result.rationale_per_arm.get(arm_id, "No rationale provided")
                    logger.info("  Selected: %s - %s", arm_id, rationale)
                else:
                    logger.warning("  Arm %s not found in pool, skipping", arm_id)

            if len(selected) < k:
                remaining = [a for a in arm_pool if a not in selected]
                additional = random.sample(remaining, min(k - len(selected), len(remaining)))
                selected.extend(additional)
                logger.warning("Added %d random arms to reach k=%d", len(additional), k)

            return selected[:k], self.current_policy

        except Exception as exc:
            logger.error("LLM arm selection failed: %s", exc)
            logger.warning("Falling back to random selection")
            return random.sample(arm_pool, k), self.current_policy


class UCBArmSelector(ArmSelector):
    """UCB (Upper Confidence Bound) arm selection.

    UCB = mean_reward + c * sqrt(ln(total_iterations) / n_trials)
    """

    def __init__(
        self,
        history: Optional[ArmHistory] = None,
        exploration_param: float = 1.0,
        target_predicates: Optional[List[str]] = None,
        relation_texts: Optional[Dict[str, str]] = None,
        entity_texts: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(
            history,
            target_predicates=target_predicates,
            relation_texts=relation_texts,
            entity_texts=entity_texts,
        )
        self.c = exploration_param

    def _ucb_score(self, total_iterations: int, stats: Optional[ArmStatistics]) -> float:
        if stats is None or stats.total_iterations == 0:
            return float("inf")
        exploration_bonus = self.c * math.sqrt(math.log(total_iterations + 1) / stats.total_iterations)
        return stats.mean_reward + exploration_bonus

    def select_arms(
        self,
        arm_pool: List[ArmWithId],
        k: int = 3,
        iteration: int = 0,
    ) -> tuple[List[ArmWithId], Optional[str]]:
        if len(arm_pool) <= k:
            return arm_pool, None

        all_stats = self.history.get_all_arm_statistics()
        scored: List[tuple[ArmWithId, float]] = []

        for awi in arm_pool:
            stats = all_stats.get(awi.arm_id)
            score = self._ucb_score(iteration, stats)
            scored.append((awi, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [a for a, _ in scored[:k]]
        logger.info("Selected %d arms using UCB (c=%.3f)", len(selected), self.c)
        return selected, None


class EpsilonGreedyArmSelector(ArmSelector):
    """Epsilon-greedy arm selection."""

    def __init__(
        self,
        history: Optional[ArmHistory] = None,
        epsilon: float = 0.1,
        target_predicates: Optional[List[str]] = None,
        relation_texts: Optional[Dict[str, str]] = None,
        entity_texts: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(
            history,
            target_predicates=target_predicates,
            relation_texts=relation_texts,
            entity_texts=entity_texts,
        )
        self.epsilon = epsilon

    def select_arms(
        self,
        arm_pool: List[ArmWithId],
        k: int = 3,
        iteration: int = 0,
    ) -> tuple[List[ArmWithId], Optional[str]]:
        if len(arm_pool) <= k:
            return arm_pool, None

        all_stats = self.history.get_all_arm_statistics()
        selected: List[ArmWithId] = []

        for _ in range(k):
            available = [a for a in arm_pool if a not in selected]
            if not available:
                break

            if random.random() < self.epsilon:
                selected.append(random.choice(available))
                continue

            best_arm = None
            best_score = float("-inf")
            for awi in available:
                stats = all_stats.get(awi.arm_id)
                score = stats.mean_reward if stats is not None else 0.0
                if score > best_score:
                    best_score = score
                    best_arm = awi

            if best_arm is not None:
                selected.append(best_arm)

        logger.info("Selected %d arms using ε-greedy (ε=%.3f)", len(selected), self.epsilon)
        return selected, None


class RandomArmSelector(ArmSelector):
    """Random arm selector."""

    def select_arms(
        self,
        arm_pool: List[ArmWithId],
        k: int = 3,
        iteration: int = 0,
    ) -> tuple[List[ArmWithId], Optional[str]]:
        if len(arm_pool) <= k:
            return arm_pool, None
        return random.sample(arm_pool, k), None


def create_arm_selector(
    strategy: str = "llm_policy",
    history: Optional[ArmHistory] = None,
    **kwargs,
) -> ArmSelector:
    """Factory function to create an ArmSelector.

    Args:
        strategy: Strategy name ('llm_policy', 'ucb', 'epsilon_greedy', 'random').
        history: ArmHistory instance.
        **kwargs: Strategy-specific parameters.

    Returns:
        ArmSelector: Selector instance.
    """

    if strategy == "llm_policy":
        return LLMPolicyArmSelector(history=history, **kwargs)
    if strategy == "ucb":
        return UCBArmSelector(history=history, **kwargs)
    if strategy == "epsilon_greedy":
        return EpsilonGreedyArmSelector(history=history, **kwargs)
    if strategy == "random":
        return RandomArmSelector(history=history)

    raise ValueError(
        f"Unknown arm selector strategy: {strategy}. "
        "Supported strategies: 'llm_policy', 'ucb', 'epsilon_greedy', 'random'"
    )
