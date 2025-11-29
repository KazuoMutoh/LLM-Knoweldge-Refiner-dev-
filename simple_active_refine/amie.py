

from __future__ import annotations
import os
import csv
import json
import re
import subprocess
import tempfile
import shutil
import logging
import pickle
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, Dict, Any

# LLM関連
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, conint, confloat

from settings import OPENAI_API_KEY, PATH_AMIE_JAR

# OPENAI_API_KEYの設定
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def write_triples_tsv(triples: Iterable[Triple], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")

@dataclass(frozen=True)
class TriplePattern:
    s: str
    p: str
    o: str

    def variables(self) -> Set[str]:
        return {v for v in (self.s, self.p, self.o) if v.startswith("?")}

    def instantiate(self, theta: Dict[str, str]) -> Triple:
        """Instantiate the pattern with substitution theta (must ground all vars)."""
        def subst(x: str) -> str:
            return theta[x] if x.startswith("?") else x
        return (subst(self.s), subst(self.p), subst(self.o))
    
    def to_tuple(self):
        return (self.s, self.p, self.o)
    
    def to_tsv(self):
        return f'{self.s}\t{self.p}\t{self.o}'

@dataclass
class AmieRule:
    head: TriplePattern
    body: List[TriplePattern]
    support: Optional[float]
    std_conf: Optional[float]
    pca_conf: Optional[float]
    head_coverage: Optional[float]
    body_size: Optional[float]
    pca_body_size: Optional[float]
    raw: str
    metadata: Optional[dict] = field(default_factory=dict)


class AllRankItem(BaseModel):
    id: conint(ge=0) = Field(..., description="Global index of the rule in the original list.")
    description: str = Field(..., description="与えられたHornルールの解釈．")
    score: confloat(ge=0, le=100) = Field(..., description="ルールが意味的に重要であるか．0から100までの数値で，意味的に重要である場合は大きな値を，重要でない場合は小さな値をつける")
    reason: str = Field(..., description="scoreの値の理由．詳しく書いて．")


class AllRank(BaseModel):
    items: List[AllRankItem] = Field(..., description="Scores for ALL input rules. Do not drop any id.")


class LLMRuleFilter:
    """Ranks AMIE+ Horn rules for usefulness using an LLM.

    Args:
        model (str): Model name for the LLM.
        temperature (float): Sampling temperature.
        request_timeout (int): Timeout for LLM requests.
        max_tokens (int): Maximum tokens for LLM output.
        system_criteria (Optional[str]): Custom system prompt for ranking criteria.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        request_timeout: int = 120,
        max_tokens: int = 16000,
        system_criteria: Optional[str] = None,
    ):
        logging.info(f"Initializing LLMRuleFilter with model={model}, temperature={temperature}")
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            timeout=request_timeout,
            max_tokens=max_tokens,
        )
        self.system_criteria = system_criteria or (
            "あなたは知識グラフと論理ルールの専門家です。以下にAMIE+で抽出されたHornルールの一覧を与えます。"
            "各ルールは「Body ⇒ Head」の形式で表され、統計的な指標（support, std_conf, pca_conf, head_coverage など）も付与されています。"
            "ただし、これらの指標は単なる統計的共起に基づいているため、意味的に本当に重要であるかは保証されません。"
        )
        self.structured = self.llm.with_structured_output(AllRank)
        logging.debug("LLMRuleFilter initialized successfully.")

    @staticmethod
    def _compact_rule(rule: AmieRule, idx: int) -> Dict[str, Any]:
        """Convert an AmieRule to a compact dictionary for LLM input.

        Args:
            rule (AmieRule): The rule to convert.
            idx (int): Global index of the rule.

        Returns:
            Dict[str, Any]: Compact representation of the rule.
        """
        
        return {
            "id": idx,
            "head": rule.head.to_tsv(),
            "body": '\t'.join([body.to_tsv() for body in rule.body]),
            "support": rule.support,
            "std_conf": rule.std_conf,
            "pca_conf": rule.pca_conf,
            "head_coverage": rule.head_coverage,
            "body_size": rule.body_size,
            "pca_body_size": rule.pca_body_size,
        }

    def _messages(self, items_json: str, n_items: int) -> List:
        """Create system and human messages for LLM input.

        Args:
            items_json (str): JSON string of all rules.
            n_items (int): Number of rules.

        Returns:
            List: List of messages for the LLM.
        """
        logging.debug(f"Preparing messages for LLM with {n_items} items.")
        system = SystemMessage(
            content=(
                self.system_criteria
                + "**タスク**"
                "与えられたルールを意味的に評価し、以下の基準に基づいてスコアを0～100してください。"
                "- 意味的に妥当か（関係が実世界の知識として合理的か）妥当であれば高いスコアをつける"
                "- 知識グラフにとって有用な補完知識を提供できそうか．提供できそうであれば高いスコアをつける"
                "- 単なる偶然の共起にすぎない可能性が高いルールは低いスコアをつける"
                "- gardening_hintが含まれるrelationを含むルールのスコアは0にする"
                "**手順**"
                "以下の手順に従ってルールのスコアをつけてください"
                "Step-1: 各ルールを理解し，説明する"
                "Step-2: 各ルールにスコアを付与し，その計算の根拠を説明する"
                "Step-3: 各ルールに付与したスコアを再度確認し，相対的にスコアが妥当か検証し，必要があれば修正する"
                f"IMPORTANT: Output scores for ALL {n_items} input ids. "
                "Return ONLY the structured result."
            )
        )
        human = HumanMessage(
            content=(
                "Here are ALL rules as JSON. Score every id (no omissions):\n"
                + items_json
            )
        )
        return [system, human]

    def filter(
        self,
        rules: List[AmieRule],
        min_pca_conf: Optional[float] = None,
        min_head_coverage: Optional[float] = None,
        top_k: int = 50,
    ) -> List[AmieRule]:
        """Evaluate and rank rules using an LLM.

        Args:
            rules (List[AmieRule]): List of rules to evaluate.
            min_pca_conf (Optional[float]): Minimum PCA confidence for pre-filtering.
            min_head_coverage (Optional[float]): Minimum head coverage for pre-filtering.
            top_k (int): Number of top rules to return.

        Returns:
            List[AmieRule]: Ranked and filtered rules with LLM metadata.
        """
        logging.info(f"Filtering {len(rules)} rules with min_pca_conf={min_pca_conf}, min_head_coverage={min_head_coverage}, top_k={top_k}")

        # 1) Numeric pre-filter
        pre: List[Tuple[int, AmieRule]] = []
        for i, r in enumerate(rules):
            if min_pca_conf is not None and r.pca_conf is not None and r.pca_conf < min_pca_conf:
                logging.debug(f"Rule {i} filtered out by min_pca_conf: {r.pca_conf}")
                continue
            if min_head_coverage is not None and r.head_coverage is not None and r.head_coverage < min_head_coverage:
                logging.debug(f"Rule {i} filtered out by min_head_coverage: {r.head_coverage}")
                continue
            pre.append((i, r))
        if not pre:
            logging.warning("No rules left after numeric pre-filter.")
            return []

        # 2) Batch scoring with LLM
        items = [self._compact_rule(r, idx=i) for i, r in pre]
        items_json = json.dumps(items, ensure_ascii=False)
        msgs = self._messages(items_json, n_items=len(items))
        logging.info(f"Sending {len(items)} rules to LLM for scoring.")

        try:
            result: AllRank = self.structured.invoke(msgs)
            logging.info("LLM scoring completed successfully.")
        except Exception as e:
            logging.error(f"LLM scoring failed: {e}")
            raise

        # 3) Fill in missing responses for stability
        scored: Dict[int, Tuple[float, str]] = {it.id: (float(it.score), it.description.strip(), it.reason.strip()) for it in result.items}
        expected_ids = {i for i, _ in pre}
        missing_ids = expected_ids - set(scored.keys())
        if missing_ids:
            logging.warning(f"Missing scores for ids: {missing_ids}")
        for mid in sorted(missing_ids):
            scored[mid] = (0.0, "missing_from_llm")

        # 4) Select top rules and add metadata
        ranked = sorted(scored.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
        out: List[AmieRule] = []
        for global_idx, (score, description, reason) in ranked:
            rule = rules[global_idx]
            md = rule.metadata or {}
            md['description'] = description
            md["llm_score"] = score
            md["extracted_reason"] = reason

            rule.metadata = md
            out.append(rule)
            logging.debug(f"Rule {global_idx} ranked with score={score}, reason={reason}")
        logging.info(f"Returning top {len(out)} rules after LLM ranking.")
        return AmieRules(out)


class AmieRules:
    """A collection of AMIE rules with utility methods."""
    def __init__(self, rules: List[AmieRule] = []):
        self.rules = rules

    @staticmethod
    def _tokenize_pattern_cell(cell: str) -> List[str]:
        # CSV cell can contain multiple spaces and tabs; normalize by splitting on whitespace
        return cell.strip().split()

    @staticmethod
    def _parse_head_to_pattern(cell: str) -> TriplePattern:
        toks = AmieRules._tokenize_pattern_cell(cell)
        if len(toks) != 3:
            raise ValueError(f"Head must have exactly 3 tokens, got {len(toks)}: {toks}")
        return TriplePattern(*toks)
        
    @staticmethod
    def _split_body_tokens_to_patterns(tokens: List[str]) -> List[TriplePattern]:
        if len(tokens) % 3 != 0:
            raise ValueError(f"Body token length must be multiple of 3, got {len(tokens)}: {tokens}")
        patterns = []
        for i in range(0, len(tokens), 3):
            s, p, o = tokens[i], tokens[i+1], tokens[i+2]
            patterns.append(TriplePattern(s, p, o))
        return patterns

    def filter_by_head_relation(self, relation: str) -> "AmieRules":
        rel_name = relation
        if "(" in relation and ")" in relation:
            rel_name = relation.split("(", 1)[0].strip()
        filtered = [r for r in self.rules if rel_name in r.head]
        return AmieRules(filtered)

    def to_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "body", "head", "support", "std_conf", "pca_conf", "head_coverage",
                "body_size", "pca_body_size", "raw"
            ])
            for r in self.rules:
                writer.writerow([
                    '\t'.join([tp.to_tsv() for tp in r.body]), 
                    r.head.to_tsv(), 
                    r.support, 
                    r.std_conf, 
                    r.pca_conf, 
                    r.head_coverage,
                    r.body_size, 
                    r.pca_body_size, 
                    r.raw
                ])

    def to_dataframe(self):
        data = []
        for r in self.rules:
            data.append({
                "body": '\t'.join([tp.to_tsv() for tp in r.body]),
                "head": r.head.to_tsv(),
                "support": r.support,
                "std_conf": r.std_conf,
                "pca_conf": r.pca_conf,
                "head_coverage": r.head_coverage,
                "body_size": r.body_size,
                "pca_body_size": r.pca_body_size,
                "raw": r.raw,
                **(r.metadata or {})
            })
        return pd.DataFrame(data)

                                                                        
    @staticmethod
    def from_csv(path: str) -> "AmieRules":
        rules: List[AmieRule] = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                body_cell = row["body"]
                head_cell = row["head"]
                body_tokens = AmieRules._tokenize_pattern_cell(body_cell)
                body_patterns = AmieRules._split_body_tokens_to_patterns(body_tokens)
                head_pattern = AmieRules._parse_head_to_pattern(head_cell)
                rules.append(
                    AmieRule(
                        body=body_patterns,
                        head=head_pattern,
                        support=float(row["support"]) if row["support"] else None,
                        std_conf=float(row["std_conf"]) if row["std_conf"] else None,
                        pca_conf=float(row["pca_conf"]) if row["pca_conf"] else None,
                        head_coverage=float(row["head_coverage"]) if row["head_coverage"] else None,
                        body_size=float(row["body_size"]) if row["body_size"] else None,
                        pca_body_size=float(row["pca_body_size"]) if row["pca_body_size"] else None,
                        raw=row["raw"]
                    )
                )
        return AmieRules(rules)

    def filter(
        self,
        min_pca_conf: float = None,
        min_head_coverage: float = None,
        sort_by: str = 'llm',
        top_k = 50,
        **llm_kwargs,
    ):
        filtered_rules = []
        for rule in self.rules:
            if min_pca_conf is not None and rule.pca_conf is not None and rule.pca_conf < min_pca_conf:
                continue
            if min_head_coverage is not None and rule.head_coverage is not None and rule.head_coverage < min_head_coverage:
                continue
            filtered_rules.append(rule)

        if sort_by != 'llm':
            # llm_score以外でソートする場合
            if sort_by not in {'support', 'std_conf', 'pca_conf', 'head_coverage', 'body_size', 'pca_body_size'}:
                raise ValueError(f'Unsupported sort_by: {sort_by}')
            # sort_byでソートしてtop_kを返す
            sorted_rules = sorted(
                filtered_rules,
                key=lambda r: getattr(r, sort_by) if getattr(r, sort_by) is not None else -1,
                reverse=True
            )
            return AmieRules(sorted_rules[:top_k])
        else:
            llm_filter = LLMRuleFilter(**llm_kwargs)

            return llm_filter.filter(
                filtered_rules,
                min_pca_conf=None,
                min_head_coverage=None,
                top_k=top_k,
            )

    def to_markdown_list(self) -> str:
        """
        write rules to a markdown list
        Returns: str:
        """
        md_text = ""
        for rule in self.rules:
            body_str = ' , '.join([tp.to_tsv() for tp in rule.body])
            head_str = f'{rule.head.to_tsv()}'
            md_text += f'- {body_str}  =>  {head_str}\n'

        return md_text
    
    @staticmethod
    def run_amie(
            triples: List[Triple],
            amie_jar: Optional[str] = PATH_AMIE_JAR,
            workdir: Optional[str] = None,
            min_support: Optional[int] = None,
            min_head_coverage: Optional[float] = None,
            min_pca: Optional[float] = None,
            extra_args: Optional[List[str]] = None,
            java_opts: Optional[List[str]] = None,
            timeout_sec: int = 0,
            ) -> List[str]:
        
        # write triples to a temporary tsv file
        if workdir is None:
            tmpdir = tempfile.mkdtemp(prefix="amie_run_")
            workdir = tmpdir
        Path(workdir).mkdir(parents=True, exist_ok=True)
        tsv_path = os.path.join(workdir, "triples.tsv")
        write_triples_tsv(triples, tsv_path)

        # run AMIE
        cmd = ["java"]
        if java_opts:
            cmd.extend(java_opts)
        cmd.extend(["-jar", amie_jar, tsv_path])
        if min_support is not None:
            cmd.extend(["-mins", str(min_support)])
        if min_head_coverage is not None:
            cmd.extend(["-minhc", str(min_head_coverage)])
        if min_pca is not None:
            cmd.extend(["-minpca", str(min_pca)])
        if extra_args:
            cmd.extend(extra_args)
        logging.info(f"Running AMIE: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            out, err = proc.communicate(timeout=None if timeout_sec <= 0 else timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise
        if proc.returncode != 0:
            logging.error(f"AMIE failed (code={proc.returncode}). stderr:\n{err}")
            raise RuntimeError(f"AMIE failed (code={proc.returncode}). stderr:\n{err}")
        logging.info(f"AMIE finished successfully. Output lines: {len(out.splitlines())}")
        
        lines = out.splitlines()

        # parse rules
        rules: List[AmieRule] = []
        for ln in lines:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if '=>' not in ln:
                continue
            body_part, head_meta_part = ln.split('=>', 1)
            body = body_part.strip()
            head_meta_split = head_meta_part.strip().split('\t')
            head = head_meta_split[0].strip() if head_meta_split else ''
            head_cov = std_conf = pca_conf = support = body_size = pca_body_size = None
            if len(head_meta_split) > 1:
                try:
                    head_cov = float(head_meta_split[1])
                except Exception:
                    pass
            if len(head_meta_split) > 2:
                try:
                    std_conf = float(head_meta_split[2])
                except Exception:
                    pass
            if len(head_meta_split) > 3:
                try:
                    pca_conf = float(head_meta_split[3])
                except Exception:
                    pass
            if len(head_meta_split) > 4:
                try:
                    support = float(head_meta_split[4])
                except Exception:
                    pass
            if len(head_meta_split) > 5:
                try:
                    body_size = float(head_meta_split[5])
                except Exception:
                    pass
            if len(head_meta_split) > 6:
                try:
                    pca_body_size = float(head_meta_split[6])
                except Exception:
                    pass
                
            body_tokens = AmieRules._tokenize_pattern_cell(body)
            body_patterns = AmieRules._split_body_tokens_to_patterns(body_tokens)
            head_pattern = AmieRules._parse_head_to_pattern(head)

            rules.append(
                AmieRule(
                    body=body_patterns, head=head_pattern, support=support,
                    std_conf=std_conf, pca_conf=pca_conf,
                    head_coverage=head_cov, body_size=body_size,
                    pca_body_size=pca_body_size, raw=ln
                )
            )

        return AmieRules(rules)
    
    def filter_rules_by_head_relation(self, 
                                      head_relation: str) -> AmieRules:
        
        rules = []
        for r in self.rules:
            if r.head.p == head_relation:
                rules.append(r)

        return AmieRules(rules)
    
    def to_pickle(self, filepath: str) -> None:
        """
        Save instance to a pickle file
        """
        with open(filepath, 'wb') as fout:
            pickle.dump(self, fout)

    def from_pickle(filepath: str) -> AmieRules:
        """
        Load instance from a pickle file
        """
        with open(filepath, 'rb') as fin:
            obj = pickle.load(fin)

        if not isinstance(obj, AmieRules):
            raise ValueError(f'Loaded object is not an instance of AmieRules: {type(obj)}') 

        return obj

