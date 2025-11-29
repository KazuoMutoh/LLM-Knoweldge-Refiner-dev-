from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Set
from collections import defaultdict

Triple = Tuple[str, str, str]


@dataclass
class KG:
    """In‑memory Knowledge Graph.

    Stores triples and adjacency for subgraph extraction and lookups.
    """

    triples: List[Triple]

    def __post_init__(self):
        self.entities: Set[str] = set()
        self.relations: Set[str] = set()
        for h, r, t in self.triples:
            self.entities.add(h); self.entities.add(t); self.relations.add(r)
        # Undirected adjacency for k‑hop searches (entity → neighbor entities)
        adj: Dict[str, Set[str]] = defaultdict(set)
        # Relation‑aware adjacency (for rule matching)
        out_adj: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        in_adj: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        for h, r, t in self.triples:
            adj[h].add(t); adj[t].add(h)
            out_adj[h][r].add(t)
            in_adj[t][r].add(h)
        self._adj = adj
        self._out_adj = out_adj
        self._in_adj = in_adj

    def contains(self, triple: Triple) -> bool:
        return triple in set(self.triples)

    def neighbors_undirected(self, e: str) -> Iterable[str]:
        return self._adj.get(e, ())

    def has_edge(self, h: str, r: str, t: str) -> bool:
        return t in self._out_adj.get(h, {}).get(r, set())

    def tails(self, h: str, r: str) -> Set[str]:
        return set(self._out_adj.get(h, {}).get(r, set()))

    def heads(self, r: str, t: str) -> Set[str]:
        return set(self._in_adj.get(t, {}).get(r, set()))