"""The scored row's own outcome must never reach a scorer."""
import pytest

from recpilot.harness.leakguard import (
    LeakageError,
    POST_IMPRESSION_FIELDS,
    assert_no_outcome_fields,
    mask_outcomes,
)
from recpilot.operators.catalog import banned_reason


def test_pre_impression_fields_pass():
    assert_no_outcome_fields(["user_id", "video_id", "author_id", "tab", "dur_bucket"])


@pytest.mark.parametrize("field", sorted(POST_IMPRESSION_FIELDS))
def test_every_outcome_field_is_rejected(field):
    with pytest.raises(LeakageError):
        assert_no_outcome_fields(["user_id", field])


def test_mask_strips_outcomes_but_keeps_context():
    row = {"user_id": "7", "video_id": "3", "duration_ms": 5000.0, "tab": "1",
           "play_time_ms": 42000.0, "is_click": 1, "long_view": 1}
    (masked,) = mask_outcomes([row])
    assert masked == {"user_id": "7", "video_id": "3", "duration_ms": 5000.0, "tab": "1"}
    assert "play_time_ms" in row, "must not mutate the caller's rows"


def test_kit_tuples_pass_through():
    row = (20220408, "1", "2", "3", "1", 5000.0, 1)
    assert mask_outcomes([row]) == [row]


def test_watch_time_operator_stays_banned():
    reason = banned_reason("add_watch_time_ranker", {})
    assert reason and "leak" in reason.lower()
