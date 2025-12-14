
from __future__ import annotations
import csv
import json
import os
import random
import shutil
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union
from collections import defaultdict
from pykeen.triples import TriplesFactory
from simple_active_refine.visualization import visualize_triples


from simple_active_refine.amie import AmieRules
from simple_active_refine.util import get_logger

logger = get_logger('triples_editor')

"""
Horn-rule body retriever for a given head triple.

Mathematical overview
---------------------
Given rules of the form
    ρ: (∧_{i=1..m} β_i(v)) ⇒ α(v)
and a concrete head triple h = (s_h, r_h, o_h),
find all substitutions θ ⊇ θ0 that satisfy the conjunctive query ∧ β_i(v) over
a candidate triple set S, where θ0 unifies α(v) with h. Return the union of
instantiated body triples {β_i θ} across all rules and substitutions.
If no match exists, return [].

Usage
-----
1) As a library:

    rules = parse_amie_csv("rules.csv")
    head = ("Alice", "/people/person/nationality", "Japan")
    # candidates can be a Python list[tuple[str,str,str]] or a TSV file path
    candidates = load_triples_tsv("triples.tsv")
    triples = find_body_triples_for_head(head, rules, candidates)

2) CLI:

    python retrieve_body_triples.py \
        --rules rules.csv \
        --head "Alice" "/people/person/nationality" "Japan" \
        --triples-tsv triples.tsv

    # or with a small inline candidate set (JSON list of triples)
    python retrieve_body_triples.py \
        --rules rules.csv \
        --head "Alice" "/people/person/nationality" "Japan" \
        --triples-json '[["USA","/location/location/contains","New York"], ...]'

Notes
-----
- The AMIE-style CSV is assumed to have columns: body, head, ... (others are ignored).
- A body cell encodes a sequence of triple patterns concatenated by spaces:
  e.g., "?x  /r1  ?y  ?z  /r2  ?x"  → two patterns: (?x,/r1,?y), (?z,/r2,?x).
- Variables start with '?' (e.g., "?a", "?b", "?f"). Constants are other strings.

"""


Triple = Tuple[str, str, str]

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

@dataclass
class Rule:
    head: TriplePattern
    body: List[TriplePattern]
    support: Optional[float]
    std_conf: Optional[float]
    pca_conf: Optional[float]
    head_coverage: Optional[float]
    body_size: Optional[float]
    pca_body_size: Optional[float]


def _tokenize_pattern_cell(cell: str) -> List[str]:
    # CSV cell can contain multiple spaces and tabs; normalize by splitting on whitespace
    return cell.strip().split()

def _split_body_tokens_to_patterns(tokens: List[str]) -> List[TriplePattern]:
    if len(tokens) % 3 != 0:
        raise ValueError(f"Body token length must be multiple of 3, got {len(tokens)}: {tokens}")
    patterns = []
    for i in range(0, len(tokens), 3):
        s, p, o = tokens[i], tokens[i+1], tokens[i+2]
        patterns.append(TriplePattern(s, p, o))
    return patterns

def _parse_head_to_pattern(cell: str) -> TriplePattern:
    toks = _tokenize_pattern_cell(cell)
    if len(toks) != 3:
        raise ValueError(f"Head must have exactly 3 tokens, got {len(toks)}: {toks}")
    return TriplePattern(*toks)

def parse_amie_csv(csv_path: str, encoding: str = "utf-8") -> List[Rule]:
    """Parse AMIE-style rules from a CSV file.

    Args:
        csv_path: Path to CSV with columns including 'body' and 'head'.
        encoding: File encoding.

    Returns:
        List of Rule objects (head pattern + body patterns).
    """
    rules: List[Rule] = []
    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            body_cell = row["body"]
            head_cell = row["head"]
            body_tokens = _tokenize_pattern_cell(body_cell)
            body_patterns = _split_body_tokens_to_patterns(body_tokens)
            head_pattern = _parse_head_to_pattern(head_cell)
            # Optionally parse additional columns if present
            support = float(row.get("support", "nan")) if "support" in row else None
            std_conf = float(row.get("std_conf", "nan")) if "std_conf" in row else None
            pca_conf = float(row.get("pca_conf", "nan")) if "pca_conf" in row else None
            head_coverage = float(row.get("head_coverage", "nan")) if "head_coverage" in row else None
            body_size = float(row.get("body_size", "nan")) if "body_size" in row else None
            pca_body_size = float(row.get("pca_body_size", "nan")) if "pca_body_size" in row else None
            rules.append(Rule(
                head=head_pattern,
                body=body_patterns,
                support=support,
                std_conf=std_conf,
                pca_conf=pca_conf,
                head_coverage=head_coverage,
                body_size=body_size,
                pca_body_size=pca_body_size
            ))
    return rules

def load_triples_tsv(tsv_path: str, encoding: str = "utf-8") -> List[Triple]:
    """Load triples from a 3-column TSV: subject \\t predicate \\t object per line."""
    triples: List[Triple] = []
    with open(tsv_path, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"TSV line must have 3 columns, got {len(parts)}: {line}")
            triples.append((parts[0], parts[1], parts[2]))
    return triples

class TripleIndex:
    """Simple indices to accelerate pattern matching on an (s,p,o) store."""
    def __init__(self, triples: Iterable[Triple]) -> None:
        self.exists: Set[Triple] = set(triples)
        self.by_sp: Dict[Tuple[str, str], List[str]] = defaultdict(list)   # (s,p) -> [o]
        self.by_po: Dict[Tuple[str, str], List[str]] = defaultdict(list)   # (p,o) -> [s]
        self.by_so: Dict[Tuple[str, str], List[str]] = defaultdict(list)   # (s,o) -> [p]
        self.by_s: Dict[str, List[Tuple[str, str]]] = defaultdict(list)    # s -> [(p,o)]
        self.by_p: Dict[str, List[Tuple[str, str]]] = defaultdict(list)    # p -> [(s,o)]
        self.by_o: Dict[str, List[Tuple[str, str]]] = defaultdict(list)    # o -> [(s,p)]
        for s,p,o in self.exists:
            self.by_sp[(s,p)].append(o)
            self.by_po[(p,o)].append(s)
            self.by_so[(s,o)].append(p)
            self.by_s[s].append((p,o))
            self.by_p[p].append((s,o))
            self.by_o[o].append((s,p))

    def match_pattern(self, pat: TriplePattern, theta: Dict[str, str]) -> Iterator[Triple]:
        """Yield all triples in the store that agree with pat under partial substitution theta."""
        # Resolve constants/variables with current substitution
        def resolve(x: str) -> Optional[str]:
            if x.startswith("?"):
                return theta.get(x)  # may be None (unbound)
            return x

        s_res, p_res, o_res = resolve(pat.s), resolve(pat.p), resolve(pat.o)

        # Use the most specific index available
        if s_res is not None and p_res is not None and o_res is not None:
            t = (s_res, p_res, o_res)
            if t in self.exists:
                yield t
            return

        if s_res is not None and p_res is not None:
            for o in self.by_sp.get((s_res, p_res), []):
                yield (s_res, p_res, o)
            return

        if p_res is not None and o_res is not None:
            for s in self.by_po.get((p_res, o_res), []):
                yield (s, p_res, o_res)
            return

        if s_res is not None and o_res is not None:
            for p in self.by_so.get((s_res, o_res), []):
                yield (s_res, p, o_res)
            return

        if s_res is not None:
            for p, o in self.by_s.get(s_res, []):
                yield (s_res, p, o)
            return

        if p_res is not None:
            for s, o in self.by_p.get(p_res, []):
                yield (s, p_res, o)
            return

        if o_res is not None:
            for s, p in self.by_o.get(o_res, []):
                yield (s, p, o_res)
            return

        # fully unbound: return everything (could be large)
        for t in self.exists:
            yield t

def _unify_constants(x: str, y: str) -> bool:
    """Return True if constants x and y are equal, or not both constants."""
    if not x.startswith("?") and not y.startswith("?"):
        return x == y
    return True

def _extend_theta_with_triple(theta: Dict[str, str],
                              pat: TriplePattern,
                              triple: Triple) -> Optional[Dict[str, str]]:
    """Attempt to extend substitution theta so that pat maps to triple."""
    s_map = {}
    for pat_sym, val in zip((pat.s, pat.p, pat.o), triple):
        if not _unify_constants(pat_sym, val):
            return None
        if pat_sym.startswith("?"):
            bound = theta.get(pat_sym)
            if bound is None:
                s_map[pat_sym] = val
            elif bound != val:
                return None
    # consistent: return extended mapping
    new_theta = dict(theta)
    new_theta.update(s_map)
    return new_theta

def _backtrack_patterns(patterns: List[TriplePattern],
                        idx: TripleIndex,
                        theta0: Dict[str, str]) -> Iterator[Tuple[Dict[str, str], List[Triple]]]:
    """Depth-first join over patterns, yielding (theta, used_triples)."""
    used: List[Triple] = []

    def dfs(i: int, theta: Dict[str, str]) -> Iterator[Tuple[Dict[str, str], List[Triple]]]:
        if i == len(patterns):
            yield theta, list(used)
            return
        pat = patterns[i]
        for t in idx.match_pattern(pat, theta):
            new_theta = _extend_theta_with_triple(theta, pat, t)
            if new_theta is None:
                continue
            used.append(t)
            yield from dfs(i+1, new_theta)
            used.pop()

    yield from dfs(0, dict(theta0))

def _unify_head_with_triple(head_pat: TriplePattern, head_triple: Triple) -> Optional[Dict[str, str]]:
    """If head pattern can unify with given head triple, return initial substitution dict."""
    theta: Dict[str, str] = {}
    for pat_sym, val in zip((head_pat.s, head_pat.p, head_pat.o), head_triple):
        if not _unify_constants(pat_sym, val):
            return None
        if pat_sym.startswith("?"):
            bound = theta.get(pat_sym)
            if bound is None:
                theta[pat_sym] = val
            elif bound != val:
                return None
        else:
            # constants must match already enforced by _unify_constants
            pass
    return theta

def find_body_triples_for_head(
    head_triple: Triple,
    amie_rules: Union[AmieRules, List[Rule]],
    candidates: Union[Sequence[Triple], TripleIndex],
) -> List[Triple]:
    """Return body triples from `candidates` that satisfy some rule for the given `head_triple`.

    The result is the deduplicated union of all instantiated body triples across
    all rules whose head unifies with `head_triple`. If nothing matches, [].

    Args:
        head_triple: (subject, predicate, object) for the head.
        rules: Parsed rules.
        candidates: Candidate triples either as a list/sequence or a prebuilt TripleIndex.

    Returns:
        List of unique triples (s,p,o) from candidates that fulfill the body.
    """
    idx = candidates if isinstance(candidates, TripleIndex) else TripleIndex(candidates)
    out_set: Set[Triple] = set()

    if isinstance(amie_rules, AmieRules):
        rules_iter = amie_rules.rules
    else:
        rules_iter = amie_rules

    for rule in rules_iter:
        # quick check: head relation must match when constant
        if not (rule.head.p.startswith("?") or rule.head.p == head_triple[1]):
            continue
        theta0 = _unify_head_with_triple(rule.head, head_triple)
        if theta0 is None:
            continue
        # backtrack over body
        for theta, used_triples in _backtrack_patterns(rule.body, idx, theta0):
            # collect all instantiated body triples (used_triples already grounded)
            out_set.update(used_triples)

    return sorted(out_set)

def randomly_select_triples(target_triple: Triple, candidate_triples: List[Triple], n: int) -> List[Triple]:
    """
    target_tripleのheadもしくはtailと接続しているtripleをcandidate_triplesからランダムにn個抽出する．
    """
    head, _, tail = target_triple
    connected = [t for t in candidate_triples if head in t or tail in t]
    return random.sample(connected, min(n, len(connected)))


def add_triples_for_single_rule(dir_triples: str,
                                rule: Union[AmieRules, List[Rule]],
                                target_triples: List[Triple],
                                f_candidate_triples: Optional[str] = None,
                                f_org_triples: Optional[str] = None) -> Tuple[List[Triple], Dict]:
    """単一ルールを用いて指定されたtarget tripleセットに対してトリプルを追加
    
    Args:
        dir_triples: トリプルデータディレクトリ
        rule: 適用するルール（単一AmieRuleまたは単一要素のAmieRules/List）
        target_triples: 対象となるトリプルのリスト
        f_candidate_triples: 候補トリプルファイルパス（Noneの場合はdir_triples/train_removed.txt）
        f_org_triples: 元トリプルファイルパス（Noneの場合はdir_triples/train.txt）
    
    Returns:
        Tuple[List[Triple], Dict]: (追加されたトリプルのリスト, 詳細情報の辞書)
    """
    # ファイルパス設定
    if f_candidate_triples is None:
        f_candidate_triples = os.path.join(dir_triples, 'train_removed.txt')
    if not os.path.exists(f_candidate_triples):
        raise Exception(f'{f_candidate_triples} cannot be found in {dir_triples}')
    
    if f_org_triples is None:
        f_org_triples = os.path.join(dir_triples, 'train.txt')
    if not os.path.exists(f_org_triples):
        raise Exception(f'{f_org_triples} cannot be found in {dir_triples}.')
    
    # ルールの正規化（単一ルールに変換）
    if isinstance(rule, AmieRules):
        if len(rule.rules) != 1:
            raise ValueError(f"Expected single rule, got {len(rule.rules)} rules")
        single_rule = [rule.rules[0]]
    elif isinstance(rule, list):
        if len(rule) != 1:
            raise ValueError(f"Expected single rule, got {len(rule)} rules")
        single_rule = rule
    else:
        # AmieRule単体の場合
        single_rule = [rule]
    
    logger.info(f'Applying single rule to {len(target_triples)} target triples')
    
    # 候補トリプルの読み込み
    tf = TriplesFactory.from_path(f_candidate_triples)
    set_candidate_triples = set(map(tuple, tf.triples.tolist()))
    logger.info(f'Number of candidate triples: {len(set_candidate_triples)}')
    
    # 元トリプルの読み込み
    tf = TriplesFactory.from_path(f_org_triples)
    set_org_triples = set(map(tuple, tf.triples.tolist()))
    logger.info(f'Number of original triples: {len(set_org_triples)}')
    
    # トリプル追加処理
    triples_to_be_added_by_target = []
    set_triples_to_be_added = set()
    
    for triple in tqdm(target_triples, desc="Processing target triples"):
        logger.debug(f'Processing triple: {triple} ...')
        _triples_to_be_added = find_body_triples_for_head(triple, single_rule, set_candidate_triples)
        
        triples_to_be_added_by_target.append({
            'target_triple': triple,
            'triples_to_be_added': _triples_to_be_added
        })
        set_triples_to_be_added |= set(_triples_to_be_added)
    
    # 関連トリプルの追加（イテレーション前にコピーを作成）
    triples_to_check = list(set_triples_to_be_added)
    for triple in triples_to_check:
        h, r, t = triple
        related_triples = [tr for tr in set_candidate_triples if h in tr or t in tr]
        set_triples_to_be_added |= set(related_triples)
    
    logger.info(f'{len(set_triples_to_be_added)} triples will be added.')
    
    # 詳細情報
    details = {
        'target_triples': target_triples,
        'added_triples_by_target': triples_to_be_added_by_target,
        'total_added': len(set_triples_to_be_added),
        'num_targets': len(target_triples)
    }
    
    return list(set_triples_to_be_added), details


# TODO クラスとして実装する
def add_triples_based_on_rules(dir_triples:str, 
                               dir_updated_triples:str,
                               rules:Union[AmieRules, List[Rule]], 
                               f_target_triples=None, 
                               f_candidate_triples=None,
                               f_org_triples=None,
                               n_add=10):
    
    if f_target_triples is None:
        f_target_triples = os.path.join(dir_triples, 'target_triples.txt')
    if not os.path.exists(f_target_triples):
        raise Exception(f'{f_target_triples} cannot be found in {dir_triples}.')
    
    if f_candidate_triples is None:
        f_candidate_triples = os.path.join(dir_triples, 'train_removed.txt')
    if not os.path.exists(f_candidate_triples):
        raise Exception(f'{f_candidate_triples} cannot be found in {dir_triples}')
    
    if f_org_triples is None:
        f_org_triples = os.path.join(dir_triples, 'train.txt')
    if not os.path.exists(f_org_triples):
        raise Exception(f'{f_org_triples} cannot be found in {dir_triples}.')
    
    os.makedirs(dir_updated_triples, exist_ok=True)
    os.makedirs(os.path.join(dir_updated_triples, 'viz'),exist_ok=True)
    
    # ---------------------------------------------
    # 準備
    # ---------------------------------------------
    # 抽出されたルール
    msg = 'rules for adding triples are ...\n'
    if isinstance(rules, AmieRules):
        for rule in rules.rules:
            msg += f'{rule.body} -->  {rule.head}\n'
    else:
        for rule in rules:
            msg += f'{rule.body} -->  {rule.head}\n'
    logger.info(msg)

    # 対象となるトリプル
    list_target_triples = []
    with open(f_target_triples, 'r') as fin:
        for row in fin:
            words = row.rstrip().split('\t')
            h = words[0]
            r = words[1]
            t = words[2]
            list_target_triples.append((h,r,t))
    
    # 対象となるトリプルをランダムにn_add個選ぶ
    if n_add is not None and n_add < len(list_target_triples):
        list_target_triples = random.sample(list_target_triples, n_add)

    logger.debug(f'target triple is {list_target_triples}')
        
    # 追加候補のトリプル
    tf = TriplesFactory.from_path(f_candidate_triples)
    set_candidate_triples = set(map(tuple, tf.triples.tolist()))
    logger.info(f'Number of candiate triples: {len(set_candidate_triples)}') 

    # トリプル追加前のトリプル
    tf = TriplesFactory.from_path(f_org_triples)
    set_org_triples = set(map(tuple, tf.triples.tolist()))
    logger.info(f'Number of original triples: {len(set_org_triples)}') 


    # ---------------------------------------------
    # 追加すべきトリプルの抽出
    # ---------------------------------------------
    logger.info('Start finding triples to be added.')
    # find triples based on rules
    triples_to_be_added_by_target = []
    set_triples_to_be_added = set()
    for triple in tqdm(list_target_triples):
        logger.debug(f'Processing triple: {triple} ...')
        _triples_to_be_added = find_body_triples_for_head(triple, rules, set_candidate_triples)
        if len(_triples_to_be_added) > 0:
            visualize_triples({'red':[triple], 'blue':_triples_to_be_added},
                            output_html=os.path.join(dir_updated_triples, 'viz', f"{'__'.join(triple).replace('/','_')}.html"),
                            title="-->".join(triple))
        triples_to_be_added_by_target.append({'target_triple':triple, 'triples_to_be_added':_triples_to_be_added})
        set_triples_to_be_added |= set(_triples_to_be_added)
    
    # find related triples
    # Create a copy of the set to iterate over while modifying the original
    triples_copy = set(set_triples_to_be_added)
    for triple in triples_copy:
        h, r, t = triple
        related_triples = [tr for tr in set_candidate_triples if h in tr or t in tr]
        set_triples_to_be_added |= set(related_triples)
    
    set_remained_triples = set_candidate_triples - set_triples_to_be_added

    # ---------------------------------------------
    # ランダムに追加するトリプルを選ぶ（比較用）
    # ---------------------------------------------
    triples_to_be_added_rand_by_target = []
    set_triples_to_be_added_rand = set()
    for d in triples_to_be_added_by_target:
        triple = d['target_triple']
        n = len(d['triples_to_be_added'])
        _triples_to_be_added = randomly_select_triples(triple, set_candidate_triples, n)
        if len(_triples_to_be_added) > 0:
            visualize_triples({'red':[triple], 'blue':_triples_to_be_added},
                            output_html= os.path.join(dir_updated_triples, 'viz', f"random_{'__'.join(triple).replace('/','_')}.html"),
                            title="-->".join(triple))
        triples_to_be_added_rand_by_target.append({'target_triple':triple, 'triples_to_be_added':_triples_to_be_added})
        set_triples_to_be_added_rand |= set(_triples_to_be_added)


    set_updated_triples = set_org_triples | set_triples_to_be_added
    set_updated_triples_rand = set_org_triples | set_triples_to_be_added_rand

    logger.info(f'{len(set_triples_to_be_added)} triples will be added.')
    logger.info(f'org:{len(set_org_triples)} --> updated:{len(set_updated_triples)}')
    logger.info(f'{len(set_remained_triples)} triples will remain to be removed.')


    # ---------------------------------------------
    # ファイルに出力
    # ---------------------------------------------
    logger.info(f'Write out updated triples in {dir_updated_triples}')

    for filename in ['train.txt', 
                     'test.txt', 
                     'valid.txt', 
                     'train_removed.txt', 
                     'test_removed.txt', 
                     'valid_removed.txt',
                     'config_dataset.json',
                     'target_triples.txt']:
        
        f_org = os.path.join(dir_triples, filename)
        f_new = os.path.join(dir_updated_triples, filename)
        
        if not os.path.exists(f_org):
            logger.info(f'{f_org} cannot be found in {dir_triples}. skip copying.')
            continue
        
        shutil.copy(f_org, f_new)

    with open(os.path.join(dir_updated_triples, 'train.txt'), 'w') as fout:
        for h, r, t in set_updated_triples:
            fout.write(f'{h}\t{r}\t{t}\n')

    with open(os.path.join(dir_updated_triples, 'train_random.txt'), 'a') as fout:
        for h, r, t in set_updated_triples_rand:
            fout.write(f'{h}\t{r}\t{t}\n')
    
    with open(os.path.join(dir_updated_triples, 'train_removed.txt'),'w') as fout:
        for h, r, t, in set_remained_triples:
            fout.write(f'{h}\t{r}\t{t}\n')

    with open(os.path.join(dir_updated_triples, 'added_triples_by_target.json'), 'w') as fout:
        json.dump(triples_to_be_added_by_target, fout, indent=2) 

    with open(os.path.join(dir_updated_triples, 'added_triples_by_target_random.json'), 'w') as fout:
        json.dump(triples_to_be_added_rand_by_target, fout, indent=2)

        
if __name__ == "__main__":
    main()