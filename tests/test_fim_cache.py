"""Factor cache: key sensitivity and npz round-trip. No simulator needed."""

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from asid.fim.cache import ExactCache, FactorCache, digest, exact_key, factor_key
from asid.fim.sketch import Sketch


@dataclass(frozen=True)
class _Config:
    frames: int = 4
    frame_dt: float = 1.0 / 30.0


@dataclass(frozen=True)
class _Case:
    name: str = "case"
    nodes: np.ndarray = None  # type: ignore[assignment]


def _sketch(value: float = 1.5) -> Sketch:
    return Sketch(
        directions=np.eye(2),
        JQ=np.full((3, 2), value),
        T=np.eye(2) * value,
        U=np.eye(2) * value,
        gain=value,
    )


class FactorCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_round_trip(self) -> None:
        cache = FactorCache(self.root)
        self.assertIsNone(cache.get("missing"))
        cache.put("key", _sketch(2.0))
        loaded = cache.get("key")
        self.assertIsNotNone(loaded)
        self.assertAlmostEqual(loaded.gain, 2.0)
        np.testing.assert_allclose(loaded.U, np.eye(2) * 2.0)
        self.assertEqual((cache.hits, cache.misses), (1, 1))
        self.assertIn("1/2", cache.report())

    def test_disabled_cache_never_hits(self) -> None:
        cache = FactorCache(None)
        cache.put("key", _sketch())
        self.assertIsNone(cache.get("key"))
        self.assertEqual(cache.hits, 0)

    def _key(self, **overrides):
        case = _Case(nodes=np.zeros((4, 3)))
        values = dict(
            case=case,
            config=_Config(),
            phi0=np.zeros(5),
            controller=np.zeros((4, 2, 3)),
            contact=np.zeros((4, 6, 3)),
            rank=2,
            step=0.01,
            sigma=0.005,
            seed=0,
        )
        values.update(overrides)
        return factor_key(**values)

    def test_key_tracks_every_input(self) -> None:
        base = self._key()
        self.assertEqual(base, self._key())
        self.assertNotEqual(base, self._key(controller=np.ones((4, 2, 3))))
        self.assertNotEqual(base, self._key(contact=np.ones((4, 6, 3))))
        self.assertNotEqual(base, self._key(rank=3))
        self.assertNotEqual(base, self._key(step=0.02))
        self.assertNotEqual(base, self._key(seed=1))
        self.assertNotEqual(base, self._key(phi0=np.ones(5)))
        self.assertNotEqual(base, self._key(config=_Config(frames=8)))

    def test_digest_distinguishes_arrays(self) -> None:
        self.assertNotEqual(digest(np.zeros(3)), digest(np.ones(3)))
        self.assertEqual(digest(np.zeros(3)), digest(np.zeros(3)))


class ExactCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _key(self, **overrides):
        values = dict(
            case=_Case(nodes=np.zeros((4, 3))),
            config=_Config(),
            phi0=np.zeros(5),
            controller=np.zeros((4, 2, 3)),
            contact=np.zeros((4, 6, 3)),
            step=0.01,
            sigma=0.005,
        )
        values.update(overrides)
        return exact_key(**values)

    def test_round_trip_and_gain_is_scalar(self) -> None:
        cache = ExactCache(self.root)
        self.assertIsNone(cache.get("missing"))
        cache.put("key", 628.373, np.eye(3))
        cached = cache.get("key")
        self.assertIsNotNone(cached)
        self.assertAlmostEqual(cached[0], 628.373)
        np.testing.assert_allclose(cached[1], np.eye(3))
        self.assertEqual((cache.hits, cache.misses), (1, 1))
        self.assertIn("1/2", cache.report())

    def test_disabled_cache_never_hits(self) -> None:
        cache = ExactCache(None)
        cache.put("key", 1.0, np.eye(2))
        self.assertIsNone(cache.get("key"))
        self.assertEqual(cache.hits, 0)

    def test_key_ignores_rank_and_seed_but_tracks_the_action(self) -> None:
        base = self._key()
        self.assertEqual(base, self._key())
        self.assertNotEqual(base, self._key(controller=np.ones((4, 2, 3))))
        self.assertNotEqual(base, self._key(contact=np.ones((4, 6, 3))))
        self.assertNotEqual(base, self._key(step=0.02))
        self.assertNotEqual(base, self._key(sigma=0.01))
        self.assertNotEqual(base, self._key(phi0=np.ones(5)))
        self.assertNotEqual(base, self._key(config=_Config(frames=8)))
        # An exact gain has no rank or seed, so it must not be keyed like a factor.
        self.assertNotEqual(base, factor_key(**self._key_args()))

    def _key_args(self):
        return dict(
            case=_Case(nodes=np.zeros((4, 3))),
            config=_Config(),
            phi0=np.zeros(5),
            controller=np.zeros((4, 2, 3)),
            contact=np.zeros((4, 6, 3)),
            rank=2,
            step=0.01,
            sigma=0.005,
            seed=0,
        )


if __name__ == "__main__":
    unittest.main()
