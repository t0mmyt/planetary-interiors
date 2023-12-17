import csv
import os
import re
from dataclasses import dataclass, field
from functools import reduce

import numpy as np
import pandas as pd


class PeriodicTable:
    oxide_split = re.compile(r"([A-Z][a-z]*\d*)")
    element_split = re.compile(r"([A-Za-z]+)(\d*)")

    """Gives Periodic Table from https://doi.org/10.1515/cti-2020-0006"""

    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(__file__), "data/periodic.csv")
        self._pt = pd.read_csv(csv_path, index_col="Symbol")

    def sym(self, element: str) -> pd.Series:
        """Fetches attributes for a given symbol (e.g. Fe)"""
        return self._pt.loc[element]

    def molecular_mass(self, formula: str) -> np.float64:
        """Molar mass for a given formula (e.g. SiO2)"""
        counts = self.atomic_counts(formula)
        return reduce(lambda w, e: w + self.sym(e).AtomicMass * counts[e], counts, 0)

    @staticmethod
    def atomic_counts(formula: str) -> dict[str, int]:
        """Atomic counts per element for a given formula"""

        def counts(element: str):
            m = PeriodicTable.element_split.match(element)
            return m.group(1), int(m.group(2)) if m.group(2) else 1

        return dict(map(counts, PeriodicTable.oxide_split.findall(formula)))


@dataclass
class BulkComposition:
    pt: PeriodicTable = field(default_factory=PeriodicTable)
    oxides_wt_pct: dict[str, np.float64] = field(default_factory=dict)

    def add(self, oxide: str, wt_pct: float | np.float64):
        self.oxides_wt_pct[oxide] = np.float64(wt_pct)

    @property
    def total(self):
        return np.sum(list(self.oxides_wt_pct.values()))

    def molar_masses(self) -> dict[str, np.float64]:
        total = self.total
        elems = {}
        for ox, wt in self.oxides_wt_pct.items():
            oxide_prop = wt / total
            oxide_wt = self.pt.molecular_mass(ox)
            for atom, n in self.pt.atomic_counts(ox).items():
                elem_wt = oxide_prop * n * self.pt.sym(atom).AtomicMass / oxide_wt
                if atom in elems:
                    elems[atom] += elem_wt
                else:
                    elems[atom] = elem_wt
        return elems

    @classmethod
    def from_csv_model(cls, csv_file: str) -> "BulkComposition":
        """Reads a bulk silicate model from a CSV file containing 'oxide' and 'wt_pct'"""
        bs = BulkComposition()
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                bs.add(**row)
        return bs
