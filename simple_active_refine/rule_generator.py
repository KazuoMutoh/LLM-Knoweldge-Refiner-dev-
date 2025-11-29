import os
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any, Optional
import json

from settings import OPENAI_API_KEY
from simple_active_refine.amie import AmieRule, TriplePattern, AmieRules
from simple_active_refine.util import get_logger
import time
import random

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


logger = get_logger('rule generator')

class RuleStructure(BaseModel):
    head: str
    body: list[str]

class Rules(BaseModel):
    rules: List[RuleStructure]

class RulesUpdate(BaseModel):
    observation: str
    thought: str
    updated_rules: Rules


class BaseRuleGenerator:
    """
    生成AIを使って，知識グラフにトリプルを追加するためのルールを生成したり，更新したりするクラスの基底クラス
    """

    system_prompt = """
    You are a rule generator for knowledge graph refinement.
    You strictly follow the output schema, never emit natural language, and never include the target relation inside rule bodies.
    You prefer short, interpretable, high-precision Horn rules.
    """

    def __init__(self, chat_model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model_name=chat_model, temperature=0)
        self.structured_llm_for_generate = self.llm.with_structured_output(Rules)
        self.structured_llm_for_update = self.llm.with_structured_output(RulesUpdate)


    def generate_rules(self, 
                       knowledge_graph, 
                       target_relation:str, 
                       n_rules:int=10, 
                       list_relations=None,
                       ref_rules:AmieRules=None) -> List[AmieRule]:
        """
        instructionに基づいてルールを生成するメソッド
        
        Args:
            instruction (str): ルール生成のための指示
            
        Returns:
            RuleStructure: 生成されたルール
        """

        prompt = f"""
        You are a senior rule miner for knowledge graph refinement. Your job is to propose ONLY high-precision Horn rules that retrieve external evidence triples which help verify or refute triples with the target relation.

        You should refer to the ‘REFERENCE RULES,’ which are extracted by AMIE+ from triples with high knowledge graph embedding scores and their neighboring triples, if such rules are provided.

        TASK
        - Generate EXACTLY {n_rules} rules.
        - Target relation (head): ?a {target_relation} ?b
        - Knowledge graph: {knowledge_graph}

        REFERENCE RULES
        {ref_rules.to_dataframe().to_markdown() if ref_rules else "No reference rule is provided."}
        
        STRICT OUTPUT SCHEMA (JSON ONLY; no prose, no markdown):
        {{
        "rules": [
            {{ "head": "<triple>", "body": ["<triple>", "..."] }},
            ...
        ]
        }}
        - "head" MUST be the literal string: "?a {target_relation} ?b"
        - "body" MUST be a list of 1–4 triples.

        TRIPLE FORMAT (BOTH head AND body)
        - Each triple is exactly three tokens separated by a single space: "<subject> <predicate> <object>"
        - Subjects/objects are variables only, starting with "?" (e.g., ?a, ?b, ?c, ?d ...)
        - Predicates are FB15k-237 relation IRIs/paths (e.g., /people/person/place_of_birth)
        - DO NOT include quotes, commas, tabs, brackets, or extra text.

        ALLOWED VARIABLES
        - Use ?a and ?b in head.
        - In body, you may reuse ?a and ?b and introduce intermediates ?c, ?d, ?e, ... as needed.
        - Do NOT introduce any constant entity IDs (/m/...), surface forms ("USA"), or literals.

        RELATION ALLOWLIST
        - If an allowlist is provided, you MUST use ONLY relations from it in the body.
        - If no allowlist is provided, you may choose relations from the KG but still follow all other constraints.
        {("ALLOWLIST:" + ",".join(list_relations)) if list_relations else "NO EXPLICIT ALLOWLIST PROVIDED."}

        QUALITY & SAFETY CONSTRAINTS
        - DO NOT include the target relation "{target_relation}" (or its trivial inverse) in the body.
        - Avoid identity/duplicate or tautological rules (e.g., using same predicate-path twice without adding new information).
        - Prefer high-precision, semantically grounded cues (e.g., place_of_birth → country, administrative_division/country, citizenship, location/contains).
        - Keep bodies short (1–4 atoms) and interpretable.
        - Avoid overly generic hubs that frequently cause spurious matches unless combined with strong constraints.

        GOOD EXAMPLE (illustrative):
        head: ?a {target_relation} ?b
        body:
        - ?a /people/person/place_of_birth ?c
        - ?c /location/administrative_division/country ?b

        BAD EXAMPLES (NEVER DO THESE):
        - Use target relation in body:
        - ?a {target_relation} ?b
        - Use constants:
        - "USA" /location/location/contains ?c
        - Wrong tokenization or brackets:
        - ["?a /people/person/place_of_birth ?c"]
        - Non-variable subjects/objects or missing tokens:
        - ?a /people/person/place_of_birth
        - ?a place_of_birth ?c
        - Overly long bodies (>4 triples)

        EVALUATION INTENT
        - Aim for rules that are precise when used to retrieve external evidence.
        - Prefer relations that narrow down nationality/affiliation via locations, citizenship, administrative boundaries, or well-typed role/organization links.

        Return ONLY the JSON object that matches the STRICT OUTPUT SCHEMA above.
        """

        logger.info(f'Ask LLM {self.system_prompt+prompt}')

        
        max_retries = 4
        base_delay = 1.0
    
        for attempt in range(1, max_retries + 1):
            try:
                result = self.structured_llm_for_generate.invoke(self.system_prompt + prompt)
                break
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                if attempt >= max_retries:
                    logger.warning(f"Error generating rules after {attempt} attempts: {e}")
                    return []
                # exponential backoff with jitter
                sleep_time = base_delay * (2 ** (attempt - 1)) * (0.8 + random.random() * 0.4)
                logger.warning(f"Attempt {attempt}/{max_retries} failed to generate rules: {e}. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
        
        return self._to_rules(result)
    

    def update_rules(self, 
                     rules:AmieRules, 
                     report: str,
                     ref_rules:AmieRules) -> RuleStructure:
        """
        instructionに基づいてルールを更新するメソッド
        
        Args:
            instruction (str): ルール更新のための指示
            
        Returns:
            RuleStructure: 更新されたルール
        """

        df_rules = rules.to_dataframe().to_markdown()

        prompt = f"""
        You are updating an existing rule set for knowledge graph refinement.

        INPUT
        <existing_rules_as_markdown_table>
        {df_rules}
        </existing_rules_as_markdown_table>

        <embedding_score_change_report>
        {report}
        </embedding_score_change_report>


        GOAL
        - Improve precision for retrieving external evidence used to assess the SAME target relation as the existing rules' head.
        - Keep rules interpretable and short (1–4 body triples).
        - Remove or rewrite rules that appear to cause noisy/low-value evidence according to the report.
        - Prefer semantically strong paths (e.g., birth place → administrative division → country; citizenship; well-typed organization/role links).
        - DO NOT introduce constants (/m/... IDs) or literals.
        - DO NOT add the target relation (or its trivial inverse) to bodies.
        - Ensure every rule's head exactly matches the existing target pattern (e.g., "?a /people/person/nationality ?b").
        - Keep total rule count comparable unless a reduction clearly improves precision. (Anyway, do not exceed 15 rules and do not go below 3 rules.)

        OUTPUT FORMAT — JSON ONLY (no prose outside JSON)
        {{
        "observation": "<1–2 concise sentences summarizing what the report implies about the rules>",
        "thought": "<1–2 concise sentences describing the concrete edit strategy you applied>",
        "updated_rules": {{
            "rules": [
            {{ "head": "<triple>", "body": ["<triple>", "..."] }},
            ...
            ]
        }}
        }}

        TRIPLE FORMAT (for both head and body)
        - Exactly three tokens: "<subject> <predicate> <object>"
        - Subjects/objects are variables only (e.g., ?a, ?b, ?c, ...). No constants or literals.
        - Predicates are valid FB15k-237 relation IRIs/paths (e.g., /people/person/place_of_birth).
        - Bodies must have 1–4 triples.
        - Never place the target relation itself (or its trivial inverse) in any body.

        EDITING GUIDELINES
        - Drop or tighten rules implicated by the report as low-precision; merge redundant ones by keeping the higher-precision variant.
        - Small, precision-increasing tweaks are encouraged (e.g., add administrative boundary hop) while keeping body ≤ 4.
        - Keep total rule count comparable unless a reduction clearly improves precision.
        - If the embedding_score_change_report is blank, it means there no evidence triples that can be retrieved based on the rule. So, you have to think other types of triples for evidence. 
        - You should refer to the ‘REFERENCE RULES,’ which are extracted by AMIE+ from triples with high knowledge graph embedding scores and their neighboring triples, if such rules are provided.

        REFERENCE RULES
        {ref_rules.to_dataframe().to_markdown() if ref_rules else "No reference rule is provided."}
        """
        logger.info(f'Ask LLM {self.system_prompt+prompt}')
        
        max_retries = 4
        base_delay = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                result = self.structured_llm_for_update.invoke(self.system_prompt + prompt)
                break
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                if attempt >= max_retries:
                    logger.warning(f"Error generating rules after {attempt} attempts: {e}")
                    return []
                # exponential backoff with jitter
                sleep_time = base_delay * (2 ** (attempt - 1)) * (0.8 + random.random() * 0.4)
                logger.warning(f"Attempt {attempt}/{max_retries} failed to generate rules: {e}. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)

        rules = self._to_rules(result.updated_rules)
        observation = result.observation
        thought = result.thought

        return rules, observation, thought
        
    def _to_rules(self, result=None):
        
        if result is None:
            return AmieRules(rules=[])
        
        # result may be a pydantic model, dict, list, or JSON string. Normalize to a list of dicts.
        rules_raw: Optional[List[Dict[str, Any]]] = None
        # If pydantic model with attribute 'rules'
        if hasattr(result, "rules"):
            rules_raw = [r.dict() if hasattr(r, "dict") else r for r in result.rules]
        elif isinstance(result, dict) and "rules" in result:
            rules_raw = result.get("rules")
        elif isinstance(result, list):
            rules_raw = result
        elif isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict) and "rules" in parsed:
                    rules_raw = parsed["rules"]
                elif isinstance(parsed, list):
                    rules_raw = parsed
            except Exception:
                # not JSON — fall back to empty
                rules_raw = None

        if not rules_raw:
            logger.warning("No rules parsed from LLM result.")
            return AmieRules(rules=[])

        out: List[AmieRule] = []

        def parse_pattern(s: str) -> TriplePattern:
            if not isinstance(s, str):
                s = str(s)
            toks = s.strip().split()
            if len(toks) == 3:
                return TriplePattern(toks[0], toks[1], toks[2])
            # fallback: put whole string as subject and empty other fields
            return TriplePattern(s, "", "")

        for r in rules_raw:
            try:
                # r may be a pydantic model or dict-like
                if hasattr(r, "dict"):
                    rd = r.dict()
                else:
                    rd = dict(r)

                head_s = rd.get("head") or rd.get("Head") or ""
                body_v = rd.get("body") or rd.get("Body") or []
                # body might be a JSON string representation
                if isinstance(body_v, str):
                    try:
                        maybe = json.loads(body_v)
                        if isinstance(maybe, list):
                            body_v = maybe
                    except Exception:
                        # try to split if it's a whitespace-separated sequence of triples
                        body_v = body_v.strip()
                        # if enclosed in brackets, remove
                        if body_v.startswith("[") and body_v.endswith("]"):
                            body_v = [x.strip().strip("'") for x in body_v[1:-1].split(",") if x.strip()]
                        else:
                            # split into chunks of 3 tokens
                            toks = body_v.split()
                            if len(toks) % 3 == 0:
                                body_v = [" ".join(toks[i:i+3]) for i in range(0, len(toks), 3)]
                            else:
                                body_v = [body_v]

                # build TriplePattern objects
                head_tp = parse_pattern(head_s)
                body_tps = [parse_pattern(x) for x in (body_v or [])]

                amie_rule = AmieRule(
                    head=head_tp,
                    body=body_tps,
                    support=None,
                    std_conf=None,
                    pca_conf=None,
                    head_coverage=None,
                    body_size=None,
                    pca_body_size=None,
                    raw=None
                )
                out.append(amie_rule)
            except Exception as e:
                logger.warning(f"Failed to convert rule entry to AmieRule: {e}; entry={r}")
                continue

        return AmieRules(rules=out)