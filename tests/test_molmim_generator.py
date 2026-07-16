# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""Tests for the MolMIM + Hybrid generators (Phase 4). No network, no rdkit."""
import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generators.base import GenerationResult  # noqa: E402
from generators.hybrid_generator import HybridGenerator  # noqa: E402
from generators.molmim_config import MolMIMConfig, MolMIMConfigError  # noqa: E402
from generators.molmim_generator import MolMIMGenerator, extract_smiles  # noqa: E402


LOCAL = "http://localhost:8000/generate"


def local_config(**kw):
    return MolMIMConfig(base_url=LOCAL, api_key=None, backoff_seconds=0.0, **kw)


class TestExtractSmiles(unittest.TestCase):
    def test_generated_list_of_strings(self):
        self.assertEqual(extract_smiles({"generated": ["CCO", "CCN"]}), ["CCO", "CCN"])

    def test_generated_json_string(self):
        self.assertEqual(extract_smiles({"generated": '["CCO", "CCN"]'}), ["CCO", "CCN"])

    def test_molecules_object_list(self):
        body = {"molecules": [{"sample": "CCO", "score": 0.9}, {"smiles": "CCN"}]}
        self.assertEqual(extract_smiles(body), ["CCO", "CCN"])

    def test_empty(self):
        self.assertEqual(extract_smiles({}), [])
        self.assertEqual(extract_smiles(None), [])


class TestConfig(unittest.TestCase):
    def test_hosted_requires_key(self):
        cfg = MolMIMConfig(base_url="https://health.api.nvidia.com/x/generate", api_key=None)
        with self.assertRaises(MolMIMConfigError):
            cfg.require_ready()

    def test_localhost_needs_no_key(self):
        local_config().require_ready()  # must not raise

    def test_payload_clamps_and_shapes(self):
        cfg = local_config().normalise()
        payload = cfg.build_payload("CCO", 500)
        self.assertEqual(payload["num_molecules"], 100)  # capped
        self.assertEqual(payload["smi"], "CCO")
        self.assertIn("algorithm", payload)


class _FakeTransport:
    """Return queued (status, body) responses; record payloads."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append({"url": url, "headers": headers, "payload": payload})
        return self.responses.pop(0)


class TestMolMIMGenerator(unittest.TestCase):
    def test_generate_success(self):
        transport = _FakeTransport([(200, {"generated": ["CCO", "CCN", "CCO"]})])
        gen = MolMIMGenerator(config=local_config(), transport=transport, log_fn=lambda *_: None)
        result = gen.generate(target="P00918", n=3, seeds=["c1ccccc1"])
        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.source, "molmim")
        self.assertEqual(result.smiles, ["CCO", "CCN"])  # deduped
        self.assertEqual(transport.calls[0]["payload"]["smi"], "c1ccccc1")

    def test_retries_on_503_then_succeeds(self):
        slept = []
        transport = _FakeTransport([(503, {"err": "busy"}), (200, {"generated": ["CCO"]})])
        gen = MolMIMGenerator(config=local_config(max_retries=3), transport=transport,
                              sleep_fn=slept.append, log_fn=lambda *_: None)
        result = gen.generate(n=1, seeds=["CCO"])
        self.assertEqual(result.smiles, ["CCO"])
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(len(slept), 1)  # backed off once

    def test_non_retryable_400_raises(self):
        transport = _FakeTransport([(400, {"err": "bad"})])
        gen = MolMIMGenerator(config=local_config(), transport=transport, log_fn=lambda *_: None)
        with self.assertRaises(RuntimeError):
            gen.generate(n=1, seeds=["CCO"])
        self.assertEqual(len(transport.calls), 1)  # not retried

    def test_network_exception_is_retried(self):
        class Boom:
            def __init__(self):
                self.n = 0

            def __call__(self, *a, **k):
                self.n += 1
                if self.n < 2:
                    raise ConnectionError("down")
                return (200, {"generated": ["CCO"]})

        boom = Boom()
        gen = MolMIMGenerator(config=local_config(max_retries=3), transport=boom,
                              sleep_fn=lambda *_: None, log_fn=lambda *_: None)
        self.assertEqual(gen.generate(n=1, seeds=["CCO"]).smiles, ["CCO"])
        self.assertEqual(boom.n, 2)

    def test_requires_seeds(self):
        gen = MolMIMGenerator(config=local_config(), transport=_FakeTransport([]))
        with self.assertRaises(MolMIMConfigError):
            gen.generate(n=3, seeds=[])

    def test_writes_smi_file(self):
        import tempfile

        transport = _FakeTransport([(200, {"generated": ["CCO", "CCN"]})])
        gen = MolMIMGenerator(config=local_config(), transport=transport, log_fn=lambda *_: None)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "generated.smi"
            gen.generate(n=2, seeds=["CCO"], output_path=out)
            self.assertTrue(out.exists())
            self.assertIn("CCO", out.read_text())

    def test_agenerate_async(self):
        transport = _FakeTransport([(200, {"generated": ["CCO"]})])
        gen = MolMIMGenerator(config=local_config(), transport=transport, log_fn=lambda *_: None)
        result = asyncio.run(gen.agenerate(n=1, seeds=["CCO"]))
        self.assertEqual(result.smiles, ["CCO"])


class _StubGen:
    def __init__(self, name, smiles, fail=False):
        self.name = name
        self._smiles = smiles
        self._fail = fail

    def generate(self, target=None, n=30, *, seeds=None, **kw):
        if self._fail:
            raise RuntimeError(f"{self.name} down")
        return GenerationResult(smiles=self._smiles[:n], source=self.name, requested=n)


class TestHybridGenerator(unittest.TestCase):
    def test_split_5050(self):
        self.assertEqual(HybridGenerator._split(10, [0.5, 0.5], 2), [5, 5])
        self.assertEqual(HybridGenerator._split(11, [0.5, 0.5], 2), [6, 5])

    def test_merges_two_pools_and_tags_source(self):
        a = _StubGen("reinvent4", ["A1", "A2", "A3"])
        b = _StubGen("molmim", ["B1", "B2", "B3"])
        hy = HybridGenerator(generators=[a, b], log_fn=lambda *_: None)
        result = hy.generate(n=4, seeds=["CCO"])
        self.assertEqual(result.source, "hybrid")
        self.assertEqual(result.count, 4)
        sources = set(result.per_molecule_source.values())
        self.assertEqual(sources, {"reinvent4", "molmim"})

    def test_graceful_when_one_component_fails(self):
        a = _StubGen("reinvent4", ["A1", "A2", "A3", "A4"])
        b = _StubGen("molmim", [], fail=True)  # e.g. no API key
        hy = HybridGenerator(generators=[a, b], log_fn=lambda *_: None)
        result = hy.generate(n=4, seeds=["CCO"])
        self.assertEqual(result.count, 4)  # topped up from reinvent4
        self.assertTrue(all(v == "reinvent4" for v in result.per_molecule_source.values()))

    def test_all_fail_raises(self):
        a = _StubGen("reinvent4", [], fail=True)
        b = _StubGen("molmim", [], fail=True)
        hy = HybridGenerator(generators=[a, b], log_fn=lambda *_: None)
        with self.assertRaises(RuntimeError):
            hy.generate(n=4, seeds=["CCO"])


if __name__ == "__main__":
    unittest.main()
