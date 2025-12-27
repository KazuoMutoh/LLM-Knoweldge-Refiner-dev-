"""Data management helpers for iteration datasets."""

from __future__ import annotations

import os
import shutil
from typing import List, Tuple

from simple_active_refine.pipeline import RefinedKG

Triple = Tuple[str, str, str]


class IterationDataManager:
    """Manage per-iteration dataset materialization on disk."""

    def __init__(self, template_dir: str, working_dir: str) -> None:
        self.template_dir = template_dir
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)

    def write_iteration(self, iteration: int, kg: RefinedKG) -> str:
        """Write train/valid/test files for a given iteration."""
        iter_dir = os.path.join(self.working_dir, f"iter_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)

        train_path = os.path.join(iter_dir, "train.txt")
        with open(train_path, "w") as fout:
            for h, r, t in kg.triples:
                fout.write(f"{h}\t{r}\t{t}\n")

        for fname in ["valid.txt", "test.txt", "target_triples.txt", "config_dataset.json"]:
            src = os.path.join(self.template_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(iter_dir, fname))

        return iter_dir

    def write_custom(self, name: str, kg: RefinedKG) -> str:
        """Write dataset using a custom directory name under working_dir."""
        custom_dir = os.path.join(self.working_dir, name)
        os.makedirs(custom_dir, exist_ok=True)

        train_path = os.path.join(custom_dir, "train.txt")
        with open(train_path, "w") as fout:
            for h, r, t in kg.triples:
                fout.write(f"{h}\t{r}\t{t}\n")

        for fname in ["valid.txt", "test.txt", "target_triples.txt", "config_dataset.json"]:
            src = os.path.join(self.template_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(custom_dir, fname))

        return custom_dir


def load_triples(file_path: str) -> List[Triple]:
    with open(file_path, "r") as fin:
        return [tuple(line.strip().split("\t")) for line in fin if line.strip()]
