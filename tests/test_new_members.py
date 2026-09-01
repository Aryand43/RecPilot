"""Co-visitation features, simplex blend weights, and snapshot ensembling."""
import numpy as np
import pytest

from recpilot.features.counts import COVISIT_NAMES, DenseEncoder, feature_names
from recpilot.harness.leakguard import mask_outcomes
from recpilot.models.base import TrainStats, predict_snapshots
from recpilot.models.ensemble import simplex_grid
from recpilot.operators.catalog import apply_operator, official_defaults


def _row(day, u, v, lv=1):
    return {"date": day, "user_id": u, "video_id": v, "author_id": "a", "tab": "1",
            "duration_ms": 5000.0, "hourmin": 1900, "long_view": lv, "is_click": 1,
            "play_time_ms": 30000.0 if lv else 500.0}


# ---------------------------------------------------------------- co-visitation

def test_covisit_columns_are_named_and_last():
    names = feature_names()
    assert names[-2:] == list(COVISIT_NAMES)


def test_covisited_item_scores_above_zero():
    """u1 and u2 both long-viewed v1 and v2, so v2 is 'like' v1 for u3."""
    enc = DenseEncoder({"video": {}, "user": {}})
    enc.fit_train([_row(20220408, "u1", "v1"), _row(20220408, "u1", "v2"),
                   _row(20220409, "u2", "v1"), _row(20220409, "u2", "v2"),
                   _row(20220410, "u3", "v1")])
    X = enc.transform([{"date": 20220422, "user_id": "u3", "video_id": "v2",
                        "author_id": "a", "tab": "1", "duration_ms": 5000.0, "hourmin": 1900}])
    assert X[0, -2] > 0 and X[0, -1] > 0


def test_unknown_user_and_item_score_zero():
    enc = DenseEncoder({"video": {}, "user": {}})
    enc.fit_train([_row(20220408, "u1", "v1")])
    X = enc.transform([{"date": 20220422, "user_id": "stranger", "video_id": "unseen",
                        "author_id": "a", "tab": "1", "duration_ms": 5000.0, "hourmin": 1900}])
    assert list(X[0, -2:]) == [0.0, 0.0]


def test_train_row_never_covisits_with_itself():
    """The expanding window means a day-0 row sees an empty accumulator."""
    enc = DenseEncoder({"video": {}, "user": {}})
    X = enc.fit_train([_row(20220408, "u", "v")])
    assert list(X[0, -2:]) == [0.0, 0.0]


def test_covisit_needs_only_pre_impression_columns():
    enc = DenseEncoder({"video": {}, "user": {}})
    enc.fit_train([_row(20220408, "u1", "v1"), _row(20220408, "u1", "v2"),
                   _row(20220409, "u3", "v1")])
    scored = [_row(20220422, "u3", "v2")]
    assert np.isfinite(enc.transform(mask_outcomes(scored))).all()


def test_covisit_can_be_ablated_off():
    rows = [_row(20220408, "u1", "v1"), _row(20220408, "u1", "v2"), _row(20220409, "u3", "v1")]
    off = DenseEncoder({"video": {}, "user": {}}, covisit=False)
    off.fit_train(rows)
    X = off.transform([_row(20220422, "u3", "v2")])
    assert list(X[0, -2:]) == [0.0, 0.0], "toggle must zero the columns, not change the width"
    assert X.shape[1] == len(feature_names())


def test_negative_events_do_not_build_covisitation():
    """Only long-views count: a shared non-long-view says nothing about taste."""
    enc = DenseEncoder({"video": {}, "user": {}})
    enc.fit_train([_row(20220408, "u1", "v1", lv=0), _row(20220408, "u1", "v2", lv=0),
                   _row(20220409, "u3", "v1", lv=0)])
    X = enc.transform([_row(20220422, "u3", "v2")])
    assert list(X[0, -2:]) == [0.0, 0.0]


# ---------------------------------------------------------------- blend weights

@pytest.mark.parametrize("n,step,expected", [(2, 0.05, 21), (2, 0.1, 11), (3, 0.1, 66)])
def test_simplex_grid_size(n, step, expected):
    assert len(simplex_grid(n, step)) == expected


def test_simplex_weights_sum_to_one_and_are_non_negative():
    for w in simplex_grid(3, 0.1):
        assert abs(sum(w) - 1.0) < 1e-9
        assert all(x >= 0 for x in w)


def test_retired_operators_are_banned_with_their_measurement():
    """Refuted ideas stay documented and unreachable, not silently deleted."""
    for op in ("blend_add_bpr", "add_snapshot_ensemble", "add_hard_negatives_within_user"):
        with pytest.raises(ValueError, match="banned"):
            apply_operator(official_defaults(), op, {})


def test_n_member_blend_still_supports_three_members():
    """The operator is retired but the machinery it needed must keep working."""
    cfg = official_defaults()
    cfg.model.name = "blend"
    cfg.model.blend_members = ["fm", "bpr", "gbdt"]
    assert len(simplex_grid(len(cfg.model.blend_members), 0.1)) == 66


# ---------------------------------------------------------------- snapshots

class _FakeFM:
    def __init__(self):
        self.V, self.W, self.b = np.zeros(1), np.zeros(1), np.float32(0)

    def predict(self, X):
        return np.full(len(X), float(self.b))


def test_predict_snapshots_averages_and_restores_state():
    m = _FakeFM()
    m.b = np.float32(99)
    snaps = [(np.zeros(1), np.zeros(1), np.float32(1)),
             (np.zeros(1), np.zeros(1), np.float32(3))]
    out = predict_snapshots(m, np.zeros((4, 1)), snaps)
    assert list(out) == [2.0] * 4
    assert float(m.b) == 99.0, "model state must be restored after scoring"


def test_single_snapshot_is_a_plain_predict():
    m = _FakeFM()
    m.b = np.float32(7)
    assert list(predict_snapshots(m, np.zeros((2, 1)), [])) == [7.0, 7.0]


def test_snapshot_machinery_still_works_though_retired():
    cfg = official_defaults()
    cfg.model.snapshot_k = 4
    assert cfg.model.snapshot_k == 4


def test_train_stats_snapshots_default_empty():
    assert TrainStats().snapshots == []
