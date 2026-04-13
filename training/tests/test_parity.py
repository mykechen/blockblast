import numpy as np
from env.pieces import PIECE_CATALOG, get_random_pieces


def test_piece_catalog_has_25_pieces():
    assert len(PIECE_CATALOG) == 25


def test_piece_ids_match_typescript():
    expected_ids = [
        "dot", "h2", "v2", "h3", "v3", "h4", "v4", "h5", "v5",
        "sq2", "sq3", "r2x3", "r3x2",
        "L0", "L1", "L2", "L3",
        "T0", "T1", "T2", "T3",
        "S0", "S1", "Z0", "Z1",
    ]
    catalog_ids = [p.id for p in PIECE_CATALOG]
    for eid in expected_ids:
        assert eid in catalog_ids, f"Missing piece: {eid}"


def test_dot_shape():
    dot = next(p for p in PIECE_CATALOG if p.id == "dot")
    expected = np.array([[1]], dtype=np.int8)
    np.testing.assert_array_equal(dot.shape, expected)


def test_sq2_shape():
    sq2 = next(p for p in PIECE_CATALOG if p.id == "sq2")
    expected = np.array([[1, 1], [1, 1]], dtype=np.int8)
    np.testing.assert_array_equal(sq2.shape, expected)


def test_L0_shape():
    L0 = next(p for p in PIECE_CATALOG if p.id == "L0")
    expected = np.array([[1, 0], [1, 0], [1, 1]], dtype=np.int8)
    np.testing.assert_array_equal(L0.shape, expected)


def test_T2_shape():
    T2 = next(p for p in PIECE_CATALOG if p.id == "T2")
    expected = np.array([[0, 1, 0], [1, 1, 1]], dtype=np.int8)
    np.testing.assert_array_equal(T2.shape, expected)


def test_S0_shape():
    S0 = next(p for p in PIECE_CATALOG if p.id == "S0")
    expected = np.array([[0, 1, 1], [1, 1, 0]], dtype=np.int8)
    np.testing.assert_array_equal(S0.shape, expected)


def test_get_random_pieces_seeded_reproducibility():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    pieces1 = get_random_pieces(3, rng1)
    pieces2 = get_random_pieces(3, rng2)
    assert [p.id for p in pieces1] == [p.id for p in pieces2]


def test_get_random_pieces_returns_clones():
    rng = np.random.default_rng(99)
    pieces = get_random_pieces(3, rng)
    pieces[0].shape[0, 0] = 99
    catalog_piece = next(p for p in PIECE_CATALOG if p.id == pieces[0].id)
    assert catalog_piece.shape[0, 0] != 99
