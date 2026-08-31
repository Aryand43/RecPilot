"""Ensemble scorers: rank normalisation, operator wiring, and label-free scoring."""
import numpy as np
import pytest

from recpilot.harness.leakguard import mask_outcomes
from recpilot.models.ensemble import rank_within_user
from recpilot.operators.catalog import apply_operator, official_defaults


def test_rank_within_user_is_per_user_and_normalised():
    users = ["a", "a", "a", "b", "b"]
    scores = np.array([0.1, 0.9, 0.5, 3.0, 1.0])
    r = rank_within_user(users, scores)
    assert list(r[:3]) == [0.0, 1.0, 0.5]
    assert list(r[3:]) == [1.0, 0.0]


def test_rank_is_invariant_to_monotone_rescaling():
    users = ["u"] * 4
    a = rank_within_user(users, np.array([1.0, 2.0, 3.0, 4.0]))
    b = rank_within_user(users, np.array([-9.0, 0.5, 7.0, 100.0]))
    assert np.allclose(a, b)


def test_single_impression_user_gets_finite_rank():
    assert list(rank_within_user(["solo"], np.array([0.7]))) == [0.0]


def test_bag_seeds_wraps_the_parent_model_class():
    parent = official_defaults()
    parent.model.name = "listwise"
    cfg = apply_operator(parent, "bag_seeds", {"seeds": 4})
    assert (cfg.model.name, cfg.model.bag_base, cfg.model.bag_seeds) == ("seed_bag", "listwise", 4)


def test_bag_seeds_on_an_ensemble_is_a_declared_noop():
    parent = apply_operator(official_defaults(), "bag_seeds", {})
    with pytest.raises(ValueError, match="already an ensemble"):
        apply_operator(parent, "bag_seeds", {})


def test_blend_defaults_to_fitting_alpha_on_valid():
    cfg = apply_operator(official_defaults(), "blend_fm_gbdt", {})
    assert cfg.model.name == "blend" and cfg.model.blend_alpha == -1.0
    fixed = apply_operator(official_defaults(), "blend_fm_gbdt", {"alpha": 0.4})
    assert fixed.model.blend_alpha == pytest.approx(0.4)


def test_tree_features_are_computable_from_masked_rows():
    """The tree ranker must score rows whose outcome columns have been stripped."""
    from recpilot.features.counts import DenseEncoder, feature_names

    enc = DenseEncoder({"video": {}, "user": {}})
    train = [{"date": 20220408, "user_id": "u1", "video_id": "v1", "author_id": "a1",
              "tab": "1", "duration_ms": 5000.0, "hourmin": 1900,
              "long_view": 1, "is_click": 1, "play_time_ms": 20000.0}]
    assert enc.fit_train(train).shape == (1, len(feature_names()))
    scored = [{"date": 20220422, "user_id": "u1", "video_id": "v1", "author_id": "a1",
               "tab": "1", "duration_ms": 5000.0, "hourmin": 1900,
               "long_view": 1, "play_time_ms": 20000.0}]
    X = enc.transform(mask_outcomes(scored))
    assert X.shape == (1, len(feature_names())) and np.isfinite(X).all()


def test_train_rows_never_contribute_to_their_own_features():
    """Expanding window: a day-0 row sees an empty accumulator."""
    from recpilot.features.counts import DenseEncoder, feature_names

    enc = DenseEncoder({"video": {}, "user": {}})
    rows = [{"date": 20220408, "user_id": "u", "video_id": "v", "author_id": "a", "tab": "1",
             "duration_ms": 1000.0, "hourmin": 1200, "long_view": 1, "is_click": 1,
             "play_time_ms": 30000.0}]
    X = enc.fit_train(rows)
    names = feature_names()
    n_col = names.index("vid_n")
    assert X[0, n_col] == 0.0, "the row's own impression must not be counted"
