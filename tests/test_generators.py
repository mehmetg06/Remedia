# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Tests for the generator abstraction (Phase 3). No rdkit/torch required."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generators import BaseGenerator, GenerationResult, available_generators, build_generator  # noqa: E402
from generators.heuristic_generator import HeuristicGenerator  # noqa: E402
from generators.reinvent_generator import ReinventGenerator  # noqa: E402


class TestGenerationResult(unittest.TestCase):
    def test_dedup_preserves_order_and_fills_source_map(self):
        r = GenerationResult(smiles=["CCO", "CCO", "CCN", ""], source="reinvent4", requested=3)
        self.assertEqual(r.smiles, ["CCO", "CCN"])
        self.assertEqual(r.count, 2)
        self.assertEqual(r.per_molecule_source["CCO"], "reinvent4")

    def test_as_molecule_list_shape(self):
        r = GenerationResult(smiles=["CCO", "CCN"], source="molmim")
        self.assertEqual(r.as_molecule_list(), [("mol_001", "CCO"), ("mol_002", "CCN")])

    def test_per_source_counts_and_manifest(self):
        r = GenerationResult(
            smiles=["CCO", "CCN"], source="hybrid",
            per_molecule_source={"CCO": "reinvent4", "CCN": "molmim"},
            requested=2,
        )
        self.assertEqual(r.per_source_counts(), {"reinvent4": 1, "molmim": 1})
        manifest = r.to_manifest()
        self.assertEqual(manifest["source"], "hybrid")
        self.assertEqual(manifest["produced"], 2)
        self.assertEqual(manifest["per_source_counts"], {"reinvent4": 1, "molmim": 1})


class TestReinventGenerator(unittest.TestCase):
    def test_calls_installer_then_sampler_and_wraps_result(self):
        calls = {}

        def fake_installer(**kw):
            calls["install"] = kw
            return Path("/tmp/reinvent")

        def fake_sampler(**kw):
            calls["sample"] = kw
            return ["CCO", "CCN", "CCO"]  # includes a duplicate

        gen = ReinventGenerator(sampler=fake_sampler, installer=fake_installer, log_fn=lambda *_: None)
        result = gen.generate(target="P00918", n=5, seed=42, cache_dir="/cache", output_path="/out/generated.smi")

        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.source, "reinvent4")
        self.assertEqual(result.smiles, ["CCO", "CCN"])  # deduped
        self.assertEqual(result.requested, 5)
        # Underlying calls receive the pipeline's arguments.
        self.assertEqual(calls["install"]["drive_cache_dir"], "/cache")
        self.assertEqual(calls["sample"]["num_molecules"], 5)
        self.assertEqual(calls["sample"]["seed"], 42)
        self.assertEqual(calls["sample"]["output_path"], "/out/generated.smi")
        self.assertEqual(result.metadata["model"], "reinvent4.prior")

    def test_name(self):
        self.assertEqual(ReinventGenerator().name, "reinvent4")


class TestHeuristicGenerator(unittest.TestCase):
    def _funcs(self):
        return {
            "random": lambda seeds, n: [f"R{i}" for i in range(n)],
            "brics": lambda seeds, n: [f"B{i}" for i in range(n)],
            "genetic": lambda seeds, generations, population_size, docking_opts, log_fn: (
                [(f"G{i}", -7.0) for i in range(population_size)], {}),
            "fusion": lambda seeds, docking_opts, log_fn, population_size, generations: (
                [(f"F{i}", -8.0) for i in range(population_size)], {}),
        }

    def test_random_and_brics(self):
        for method, prefix in (("random", "R"), ("brics", "B")):
            gen = HeuristicGenerator(method, functions=self._funcs(), log_fn=lambda *_: None)
            result = gen.generate(seeds=["CCO"], n=4)
            self.assertEqual(result.count, 4)
            self.assertTrue(all(s.startswith(prefix) for s in result.smiles))
            self.assertEqual(result.source, f"heuristic:{method}")

    def test_genetic_and_fusion_extract_smiles(self):
        for method, prefix in (("genetic", "G"), ("fusion", "F")):
            gen = HeuristicGenerator(method, functions=self._funcs(), log_fn=lambda *_: None)
            result = gen.generate(seeds=["CCO"], n=3)
            self.assertLessEqual(result.count, 10)
            self.assertTrue(all(s.startswith(prefix) for s in result.smiles))

    def test_requires_seeds(self):
        gen = HeuristicGenerator("random", functions=self._funcs())
        with self.assertRaises(ValueError):
            gen.generate(seeds=[], n=3)

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            HeuristicGenerator("nonsense")


class TestFactory(unittest.TestCase):
    def test_default_is_reinvent(self):
        self.assertIsInstance(build_generator(), ReinventGenerator)
        self.assertIsInstance(build_generator("pretrained"), ReinventGenerator)
        self.assertIsInstance(build_generator("REINVENT"), ReinventGenerator)

    def test_heuristic_aliases(self):
        self.assertIsInstance(build_generator("fusion"), HeuristicGenerator)
        self.assertIsInstance(build_generator("heuristic:brics"), HeuristicGenerator)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            build_generator("does-not-exist")

    def test_available_list(self):
        names = available_generators()
        self.assertIn("reinvent4", names)
        self.assertIn("molmim", names)
        self.assertIn("hybrid", names)

    def test_all_generators_are_base(self):
        self.assertTrue(issubclass(ReinventGenerator, BaseGenerator))
        self.assertTrue(issubclass(HeuristicGenerator, BaseGenerator))


if __name__ == "__main__":
    unittest.main()
