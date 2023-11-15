import numpy as np
import pytest

from model.minerals import PeriodicTable, BulkSilicate


@pytest.fixture
def pt():
    return PeriodicTable()


class TestPeriodicTable:
    @pytest.mark.parametrize(
        "sym,attr,value", [("Fe", "AtomicNumber", 26), ("O", "AtomicMass", 15.999)]
    )
    def test_symbol_lookup(self, pt, sym, attr, value):
        assert pt.sym(sym)[attr] == value

    @pytest.mark.parametrize(
        "oxide,counts",
        [
            ("MnO", {"Mn": 1, "O": 1}),
            ("Cr2O3", {"Cr": 2, "O": 3}),
        ],
    )
    def test_atomic_counts(self, pt, oxide, counts):
        assert pt.atomic_counts(oxide) == counts

    @pytest.mark.parametrize(
        "oxide,wt",
        [
            ("SiO2", 60.083),
            ("MgO", 40.304),
            ("Al2O3", 101.960),
        ],
    )
    def test_oxide_mass_calculation(self, pt, oxide, wt):
        assert np.isclose(pt.molecular_mass(oxide), wt)


class TestBulkSilicate:
    bs = BulkSilicate.from_csv_model("data/test_bsm.csv")

    def test_bulk_silicate_total(self, pt):
        bs = BulkSilicate()
        oxides = [("SiO2", 45.5), ("MgO", 31.0), ("FeO", 14.7)]
        for o in oxides:
            bs.add(*o)
        assert bs.total == 91.2

    def test_bulk_silicate_from_csv_model(self):
        """Verify that summing percentages from Yoshizaki, 2020 (table 4) agrees"""
        assert np.isclose(self.bs.total, 99.939)

    @pytest.mark.parametrize(
        "oxides,pct",
        [
            (("SiO2",), (100,)),  # Qz
            (("MgO", "SiO2"), (57.294, 42.706)),  # Fo
            (("MgO", "FeO", "SiO2"), (23.401, 41.714, 34.886)),  # Ol
        ],
    )
    def test_molar_masses(self, oxides, pct):
        """Uses some common minerals to test BS Molar Masses"""
        bs = BulkSilicate()
        for i in range(len(oxides)):
            bs.add(oxides[i], pct[i])
        mm = bs.molar_masses()
        # Molar mass ratios should sum to all percentages / 100
        assert np.isclose(np.sum(pct) / 100, np.sum(list(mm.values())))
