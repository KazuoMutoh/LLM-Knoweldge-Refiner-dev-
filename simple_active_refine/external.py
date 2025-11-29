from __future__ import annotations
"""
External resource providers
---------------------------
Two provider types are implemented:
  • TriplesPool: looks up candidate missing triples in a directory containing
    TSV files with triples (e.g., the removed/held‑out pool).
  • InternetProvider: placeholder class to integrate web/Wikipedia/LLM retrieval.
"""
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Set
from pathlib import Path
from .io_utils import read_triples

Triple = Tuple[str, str, str]


@dataclass
class TriplesPool:
    dir_pool: Path

    def __post_init__(self):
        self.dir_pool = Path(self.dir_pool)
        self._index: Set[Triple] = set()
        for p in self.dir_pool.glob("*.tsv") | self.dir_pool.glob("*.txt"):
            for tr in read_triples(p):
                self._index.add(tr)

    def has(self, tr: Triple) -> bool:
        return tr in self._index


class InternetProvider:
    def find(self, tr: Triple) -> bool:
        """Placeholder for internet retrieval of a triple.

        Returns:
            Always False in this reference implementation. Extend as needed.
        """
        return False