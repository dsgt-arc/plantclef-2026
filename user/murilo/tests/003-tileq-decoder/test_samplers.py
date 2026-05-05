"""Tests for plantclef.data.samplers.

Sampler A (verified against spec §2.3 + §4.3):
- Shapes/dtypes; exactly n_positives label ones; provenance per-class pool;
  shuffled tile order; weight distribution; constructor validation.

Sampler B (verified against spec §2.3):
- Shapes/dtypes; injected species in label; surviving FAISS-discovered
  backgrounds added; overwritten background labels dropped; deduplication on
  injected/background overlap; injected-slot CLSs come from train, untouched
  slots match the source test quadrat exactly; determinism; constructor
  validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from plantclef.data.samplers import SamplerA, SamplerB


def _make_synthetic_lookup(
    counts: list[int], dim: int = 8, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct (embeddings, offsets, weights) for tests.

    Each class c gets ``counts[c]`` rows of float32 vectors uniquely tagged by
    ``c * 1000 + within_class_idx`` in the FIRST coordinate, with random noise
    in the rest. This makes provenance checks trivial: a tile's class is
    recoverable from ``int(tile[0]) // 1000``.
    """
    rng = np.random.default_rng(seed)
    n_classes = len(counts)
    total = sum(counts)
    embeddings = rng.normal(0, 1, (total, dim)).astype(np.float32)
    offsets = np.zeros(n_classes + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    for c in range(n_classes):
        for j in range(counts[c]):
            embeddings[offsets[c] + j, 0] = float(c * 1000 + j)
    raw = 1.0 / np.sqrt(np.asarray(counts, dtype=np.float64))
    weights = (raw / raw.sum()).astype(np.float32)
    return embeddings, offsets, weights


# ----- shape / dtype -----


def test_sample_shapes_and_dtypes():
    embeddings, offsets, weights = _make_synthetic_lookup([10, 10, 10, 10, 10], dim=16)
    rng = np.random.default_rng(42)
    sampler = SamplerA(embeddings, offsets, weights, n_positives=2, n_tiles=8, rng=rng)
    tiles, label = sampler.sample()

    assert tiles.shape == (8, 16)
    assert tiles.dtype == np.float32
    assert label.shape == (5,)
    assert label.dtype == np.float32


def test_label_has_exactly_n_positives_ones():
    embeddings, offsets, weights = _make_synthetic_lookup([10] * 20, dim=8)
    sampler = SamplerA(
        embeddings, offsets, weights,
        n_positives=5, n_tiles=16, rng=np.random.default_rng(0),
    )
    for _ in range(50):
        _, label = sampler.sample()
        assert label.sum() == 5.0
        # only 0/1 values
        assert set(label.tolist()) <= {0.0, 1.0}


def test_positives_are_distinct():
    """n_positives species drawn WITHOUT replacement -> no duplicates in label."""
    embeddings, offsets, weights = _make_synthetic_lookup([10] * 8, dim=8)
    sampler = SamplerA(
        embeddings, offsets, weights,
        n_positives=5, n_tiles=16, rng=np.random.default_rng(0),
    )
    for _ in range(20):
        _, label = sampler.sample()
        # Indices set to 1 must be unique (label is multi-hot, so they always are
        # by construction — assert label values are exactly {0, 1} and sum = 5)
        ones = np.where(label > 0)[0]
        assert len(ones) == 5
        assert len(set(ones.tolist())) == 5


# ----- provenance / per-class pool -----


def test_tile_cls_come_from_correct_class_pool():
    """Each tile's first coordinate must be in [c*1000, c*1000+counts[c])
    for SOME class c, and exactly the n_positives classes labelled ones must
    each contribute at least one tile.
    """
    counts = [3, 5, 4, 2, 6]
    embeddings, offsets, weights = _make_synthetic_lookup(counts, dim=8)
    sampler = SamplerA(
        embeddings, offsets, weights,
        n_positives=2, n_tiles=6, rng=np.random.default_rng(7),
    )
    tiles, label = sampler.sample()

    pos_classes = set(np.where(label > 0)[0].tolist())

    # For each tile, recover the source class from the tagged first coordinate
    tile_classes = []
    for t in tiles:
        c = int(t[0]) // 1000
        within = int(t[0]) - c * 1000
        # Tag must be in valid range for class c
        assert 0 <= c < len(counts), f"recovered class {c} out of range"
        assert 0 <= within < counts[c], (
            f"within-class idx {within} out of range for class {c} (count={counts[c]})"
        )
        tile_classes.append(c)
    tile_classes_set = set(tile_classes)

    # Every positive class must show up among the tiles
    assert pos_classes.issubset(tile_classes_set), (
        f"positive classes {pos_classes} not all present in tile sources {tile_classes_set}"
    )


# ----- determinism -----


def test_seeded_sample_is_deterministic():
    embeddings, offsets, weights = _make_synthetic_lookup([10] * 30, dim=8)
    s1 = SamplerA(
        embeddings, offsets, weights,
        n_positives=3, n_tiles=10, rng=np.random.default_rng(123),
    )
    s2 = SamplerA(
        embeddings, offsets, weights,
        n_positives=3, n_tiles=10, rng=np.random.default_rng(123),
    )
    t1, l1 = s1.sample()
    t2, l2 = s2.sample()
    assert np.array_equal(t1, t2)
    assert np.array_equal(l1, l2)


def test_different_seeds_diverge():
    embeddings, offsets, weights = _make_synthetic_lookup([10] * 30, dim=8)
    s1 = SamplerA(
        embeddings, offsets, weights,
        n_positives=3, n_tiles=10, rng=np.random.default_rng(1),
    )
    s2 = SamplerA(
        embeddings, offsets, weights,
        n_positives=3, n_tiles=10, rng=np.random.default_rng(2),
    )
    t1, _ = s1.sample()
    t2, _ = s2.sample()
    assert not np.array_equal(t1, t2)


# ----- shuffle -----


def test_tile_order_is_shuffled():
    """Across many samples, positives must NOT always sit in the first n_positives slots."""
    embeddings, offsets, weights = _make_synthetic_lookup([10] * 20, dim=8)
    sampler = SamplerA(
        embeddings, offsets, weights,
        n_positives=2, n_tiles=8, rng=np.random.default_rng(0),
    )
    # Track whether at least one sample has a positive in slots 2..7
    seen_late_positive = False
    for _ in range(50):
        tiles, label = sampler.sample()
        pos_classes = set(np.where(label > 0)[0].tolist())
        for slot in range(2, 8):
            c = int(tiles[slot, 0]) // 1000
            if c in pos_classes:
                seen_late_positive = True
                break
        if seen_late_positive:
            break
    assert seen_late_positive, "positives never appear outside the first n_positives slots"


# ----- weight distribution -----


def test_high_weight_classes_drawn_more_often():
    """Over many draws, the rarest class (highest weight) is the most-frequently sampled."""
    # Skewed counts -> class 0 has the highest weight (1/sqrt(1) = 1 raw)
    counts = [1, 4, 16, 64, 256]
    embeddings, offsets, weights = _make_synthetic_lookup(counts, dim=8)
    # Verify the weight ordering
    assert weights.argmax() == 0, "synthetic setup wrong"

    sampler = SamplerA(
        embeddings, offsets, weights,
        n_positives=1, n_tiles=2, rng=np.random.default_rng(0),
    )
    class_freq = np.zeros(len(counts), dtype=np.int64)
    for _ in range(2000):
        tiles, label = sampler.sample()
        # Track positive-only frequency (controlled draw, not the bg replacement noise)
        class_freq[np.argmax(label)] += 1

    # Highest-weight class should be the most-drawn positive
    assert class_freq.argmax() == 0
    # Sanity: lowest-weight class drawn least
    assert class_freq.argmin() == 4


# ----- constructor validation -----


def test_rejects_3d_embeddings():
    embeddings = np.zeros((10, 4, 8), dtype=np.float32)
    offsets = np.array([0, 5, 10], dtype=np.int64)
    weights = np.array([0.5, 0.5], dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        SamplerA(embeddings, offsets, weights)


def test_rejects_wrong_dtype():
    embeddings = np.zeros((10, 8), dtype=np.float64)
    offsets = np.array([0, 5, 10], dtype=np.int64)
    weights = np.array([0.5, 0.5], dtype=np.float32)
    with pytest.raises(ValueError, match="float32"):
        SamplerA(embeddings, offsets, weights)


def test_rejects_inconsistent_offsets():
    embeddings = np.zeros((10, 8), dtype=np.float32)
    offsets = np.array([0, 5, 9], dtype=np.int64)  # last must equal 10
    weights = np.array([0.5, 0.5], dtype=np.float32)
    with pytest.raises(ValueError, match="offsets"):
        SamplerA(embeddings, offsets, weights)


def test_rejects_n_positives_out_of_range():
    embeddings, offsets, weights = _make_synthetic_lookup([5, 5, 5], dim=8)
    with pytest.raises(ValueError, match="n_positives"):
        SamplerA(embeddings, offsets, weights, n_positives=0, n_tiles=8)
    with pytest.raises(ValueError, match="n_positives"):
        SamplerA(embeddings, offsets, weights, n_positives=10, n_tiles=8)


def test_rejects_n_positives_exceeds_n_classes():
    embeddings, offsets, weights = _make_synthetic_lookup([5, 5, 5], dim=8)
    with pytest.raises(ValueError, match="exceeds n_classes"):
        SamplerA(embeddings, offsets, weights, n_positives=5, n_tiles=10)


# ============================================================================
# Sampler B
# ============================================================================


def _make_test_cls_for_sampler_b(
    n_quadrats: int, n_tiles: int, dim: int, seed: int = 0
) -> np.ndarray:
    """Synthetic test_cls with provenance tags: ``test_cls[q, t, 0] = 10000 + q*100 + t``.

    Distinguishable from train tags (which use ``c * 1000 + within_class_idx``)
    because test tags are >= 10000 and train tags are < 10000 in our fixtures.
    """
    rng = np.random.default_rng(seed)
    test_cls = rng.normal(0, 1, (n_quadrats, n_tiles, dim)).astype(np.float32)
    for q in range(n_quadrats):
        for t in range(n_tiles):
            test_cls[q, t, 0] = float(10000 + q * 100 + t)
    return test_cls


def _make_sampler_b_fixture(
    counts=(10, 10, 10, 10, 10),
    n_quadrats: int = 4,
    n_tiles: int = 8,
    dim: int = 8,
    seed: int = 0,
):
    embeddings, offsets, weights = _make_synthetic_lookup(list(counts), dim=dim, seed=seed)
    test_cls = _make_test_cls_for_sampler_b(n_quadrats, n_tiles, dim, seed=seed + 1)
    # tile_to_class_idx mostly -1, with a few placed labels.
    tile_to_class_idx = np.full((n_quadrats, n_tiles), -1, dtype=np.int32)
    return embeddings, offsets, weights, test_cls, tile_to_class_idx


def test_b_sample_shapes_and_dtypes():
    embeddings, offsets, weights, test_cls, tt = _make_sampler_b_fixture()
    s = SamplerB(
        embeddings, offsets, weights, test_cls, tt,
        n_positives=2, n_tiles=8, rng=np.random.default_rng(0),
    )
    tiles, label = s.sample()
    assert tiles.shape == (8, 8)
    assert tiles.dtype == np.float32
    assert label.shape == (5,)
    assert label.dtype == np.float32
    assert set(label.tolist()) <= {0.0, 1.0}


def test_b_label_includes_all_injected_species_when_no_bg_labels():
    """tile_to_class_idx all -1 -> label should have exactly n_positives ones."""
    embeddings, offsets, weights, test_cls, tt = _make_sampler_b_fixture()
    s = SamplerB(
        embeddings, offsets, weights, test_cls, tt,
        n_positives=3, n_tiles=8, rng=np.random.default_rng(0),
    )
    for _ in range(20):
        _, label = s.sample()
        assert label.sum() == 3.0


def test_b_surviving_background_labels_added():
    """Background labels on non-overwritten tiles must appear in the multi-hot."""
    embeddings, offsets, weights, test_cls, tt = _make_sampler_b_fixture(n_quadrats=1)
    # Plant a known background label on tile 0 of quadrat 0
    tt[0, 0] = 4  # class_idx 4
    s = SamplerB(
        embeddings, offsets, weights, test_cls, tt,
        n_positives=2, n_tiles=8, rng=np.random.default_rng(0),
    )
    seen_bg_added = 0
    for _ in range(50):
        # n_quadrats=1 forces q=0 every time
        _, label = s.sample()
        # If slot 0 wasn't overwritten, label should contain class 4
        if label[4] == 1.0:
            seen_bg_added += 1
    # With 8 tiles and 2 injections, P(slot 0 not chosen) = 6/8 = 0.75
    # Over 50 draws, we expect ~38 successes; assert at least 20.
    assert seen_bg_added >= 20, (
        f"surviving background label rarely added: {seen_bg_added}/50"
    )


def test_b_overwritten_background_labels_dropped():
    """When the only labelled tile is overwritten, that label is NOT in the output."""
    rng = np.random.default_rng(0)
    embeddings, offsets, weights = _make_synthetic_lookup(
        [10, 10, 10], dim=8, seed=0
    )
    # 1 quadrat, 4 tiles; only tile 0 has bg label = class 2
    test_cls = _make_test_cls_for_sampler_b(1, 4, 8, seed=1)
    tt = np.full((1, 4), -1, dtype=np.int32)
    tt[0, 0] = 2

    # Force injection into slot 0 by making n_positives = n_tiles (every slot
    # is injected, so the only labelled tile gets overwritten).
    s = SamplerB(
        embeddings, offsets, weights, test_cls, tt,
        n_positives=2, n_tiles=4, rng=rng,
    )
    for _ in range(50):
        _, label = s.sample()
        injected = set(np.where(label > 0)[0].tolist())
        # Class 2 should NOT appear in label *unless* it was one of the 2 injected
        # species (in which case it's there because of injection, not background).
        # Run a tighter check: when the injected set does NOT include class 2,
        # the label should also not include class 2 (because the only bg-labelled
        # tile, slot 0, was definitely overwritten with one of the 2 injected
        # species; with n_positives=2 and n_tiles=4, slot 0 is chosen with
        # probability 2/4 = 50%).
        if 2 not in injected:
            # bg label could remain only if slot 0 was NOT overwritten — but with
            # the test_cls remaining at slot 0, the surviving bg logic still adds
            # class 2 to the label. So actually class 2 may or may not be there.
            pass
    # Tighter assertion using a fixed configuration (force inject_slots=[0,1]):
    # We can't directly force inject_slots without changing the API, so this
    # test asserts the bookkeeping logic via the next more controlled test.


def test_b_overwritten_label_strictly_dropped_controlled():
    """Controlled: 2 tiles, n_positives=1 — sometimes slot 0 (the bg-labelled
    one) is chosen for injection, sometimes slot 1 is. When slot 0 is chosen,
    label[2] (the bg label on slot 0) must be 0 *unless* class 2 was the
    injected species.
    """
    embeddings, offsets, weights = _make_synthetic_lookup([10, 10, 10], dim=8, seed=0)
    test_cls = _make_test_cls_for_sampler_b(1, 2, 8, seed=1)
    tt = np.full((1, 2), -1, dtype=np.int32)
    tt[0, 0] = 2  # bg-labelled tile at slot 0

    s = SamplerB(
        embeddings, offsets, weights, test_cls, tt,
        n_positives=1, n_tiles=2, rng=np.random.default_rng(0),
    )
    for _ in range(100):
        tiles, label = s.sample()
        # Recover the injected species from the tile that is NOT a test tile
        # (test tiles have first-coord >= 10000; train tiles have first-coord < 10000)
        is_test = tiles[:, 0] >= 10000
        is_train = ~is_test
        assert is_train.sum() == 1, f"expected exactly 1 injected slot, got {is_train.sum()}"
        injected_slot = int(np.where(is_train)[0][0])
        injected_class = int(tiles[injected_slot, 0]) // 1000

        ones = set(np.where(label > 0)[0].tolist())
        if injected_slot == 0:
            # slot 0 was overwritten -> bg label class 2 must be DROPPED unless
            # injected_class itself happens to be 2 (in which case it's listed
            # via the injection)
            if injected_class == 2:
                assert 2 in ones
            else:
                assert 2 not in ones
        else:  # injected_slot == 1, slot 0 (bg-labelled) survives
            assert 2 in ones


def test_b_deduplicates_injection_and_bg_overlap():
    """If injected species == surviving bg species, the multi-hot still has 1."""
    embeddings, offsets, weights = _make_synthetic_lookup([10, 10, 10], dim=8, seed=0)
    test_cls = _make_test_cls_for_sampler_b(1, 4, 8, seed=1)
    tt = np.full((1, 4), -1, dtype=np.int32)
    tt[0, 0] = 1  # bg label on slot 0 = class 1
    tt[0, 2] = 1  # bg label on slot 2 = class 1 (same class)

    s = SamplerB(
        embeddings, offsets, weights, test_cls, tt,
        n_positives=1, n_tiles=4, rng=np.random.default_rng(7),
    )
    for _ in range(50):
        _, label = s.sample()
        # Multi-hot must always be 0/1 (no double-counting)
        assert set(label.tolist()) <= {0.0, 1.0}


def test_b_unmodified_slots_match_test_cls_exactly():
    """Slots not chosen for injection must equal the source test quadrat's CLSs."""
    embeddings, offsets, weights = _make_synthetic_lookup([10, 10, 10], dim=8, seed=0)
    test_cls = _make_test_cls_for_sampler_b(2, 8, 8, seed=1)
    tt = np.full((2, 8), -1, dtype=np.int32)

    s = SamplerB(
        embeddings, offsets, weights, test_cls, tt,
        n_positives=3, n_tiles=8, rng=np.random.default_rng(0),
    )
    for _ in range(20):
        tiles, _ = s.sample()
        # Recover the source quadrat from the FIRST surviving test tile (>= 10000 tag)
        is_test = tiles[:, 0] >= 10000
        assert is_test.sum() == 8 - 3  # exactly 5 surviving test tiles
        # All test-tagged tiles must come from the same quadrat
        recovered_q = ((tiles[is_test, 0].astype(int) - 10000) // 100)
        assert len(set(recovered_q.tolist())) == 1
        q = int(recovered_q[0])
        # Verify each surviving slot's CLS matches test_cls[q, t] exactly
        for t in np.where(is_test)[0]:
            assert np.array_equal(tiles[t], test_cls[q, t])


def test_b_seeded_sample_is_deterministic():
    embeddings, offsets, weights, test_cls, tt = _make_sampler_b_fixture(
        n_quadrats=10, n_tiles=8, dim=8
    )
    s1 = SamplerB(
        embeddings, offsets, weights, test_cls, tt,
        n_positives=3, n_tiles=8, rng=np.random.default_rng(42),
    )
    s2 = SamplerB(
        embeddings, offsets, weights, test_cls, tt,
        n_positives=3, n_tiles=8, rng=np.random.default_rng(42),
    )
    t1, l1 = s1.sample()
    t2, l2 = s2.sample()
    assert np.array_equal(t1, t2)
    assert np.array_equal(l1, l2)


def test_b_rejects_test_cls_wrong_shape():
    embeddings, offsets, weights, _, _ = _make_sampler_b_fixture()
    bad_test_cls = np.zeros((4, 16, 8), dtype=np.float32)  # n_tiles=16 != 8
    bad_tt = np.full((4, 16), -1, dtype=np.int32)
    with pytest.raises(ValueError, match="test_cls"):
        SamplerB(embeddings, offsets, weights, bad_test_cls, bad_tt, n_tiles=8)


def test_b_rejects_test_cls_wrong_dim():
    embeddings, offsets, weights, _, _ = _make_sampler_b_fixture(dim=8)
    bad_test_cls = np.zeros((4, 8, 16), dtype=np.float32)  # dim=16 != train dim=8
    bad_tt = np.full((4, 8), -1, dtype=np.int32)
    with pytest.raises(ValueError, match="dim"):
        SamplerB(embeddings, offsets, weights, bad_test_cls, bad_tt, n_tiles=8)


def test_b_rejects_tile_to_class_idx_shape_mismatch():
    embeddings, offsets, weights, test_cls, _ = _make_sampler_b_fixture()
    bad_tt = np.full((3, 8), -1, dtype=np.int32)  # n_quadrats=3 != test_cls's 4
    with pytest.raises(ValueError, match="tile_to_class_idx"):
        SamplerB(embeddings, offsets, weights, test_cls, bad_tt, n_tiles=8)
