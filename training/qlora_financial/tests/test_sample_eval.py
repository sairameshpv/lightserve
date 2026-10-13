"""Scoring helpers in sample_eval.py, on made-up answers. Pure Python, no GPU."""
from training.qlora_financial.sample_eval import VARIANTS, bucket, majority_vote, match_variant, pass_at_k


def test_pass_at_k_by_hand():
    assert abs(pass_at_k(16, 4, 8) - (1 - 495 / 12870)) < 1e-12  # C(12,8)=495, C(16,8)=12870
    assert pass_at_k(16, 0, 8) == 0.0
    assert pass_at_k(16, 9, 8) == 1.0  # only 7 wrong: any 8 picks include a right one
    assert abs(pass_at_k(16, 4, 1) - 4 / 16) < 1e-12  # pass@1 = plain accuracy


def test_majority_vote():
    assert majority_vote([14.1, 14.10, 3.0, None, None, None]) == 14.1  # Nones never win
    assert majority_vote([100.0, 100.5, 7.0]) == 100.0  # within 1% = same answer
    assert majority_vote([None, None]) is None
    assert majority_vote([5.0, 6.0]) == 5.0  # tie: first seen


def test_bucket():
    assert [bucket(c, 16) for c in (0, 7, 16)] == ["never", "sometimes", "always"]


def test_match_variant():
    year = [match_variant("Answer: 2019", "2018", v) for v in VARIANTS]
    assert year == [True, False, False, False]  # within 1%, but a different year
    scale = [match_variant("$5,478,733 thousand", "$5,478,733", v) for v in VARIANTS]
    assert scale == [False, False, True, True]  # same digits, extra scale word
    near = [match_variant("Answer: 0.0665", "Answer: 0.06625", v) for v in VARIANTS]
    assert near == [True, True, True, False]  # 0.4% off: inside 1%, outside 0.1%
    swap = [match_variant("Answer: -40 million", "Answer: -40 thousand", v) for v in VARIANTS]
    assert swap == [False, False, False, False]  # swapped scale word: 1,000x off
    rounded = [match_variant("Answer: 39.34 percent", "Answer: 39.3 percent", v) for v in VARIANTS]
    assert rounded == [True, True, True, True]  # 0.1% off but honest rounding
    assert not match_variant("Answer: 0.7168", "Answer: 0.71564", "fair")  # 151/211 slip
    assert not match_variant("Answer: -1.0", "Answer: 0.0", "fair")  # -1.0/100 must not round to 0.0
    assert match_variant("Answer: 7.66%", "Answer: 0.0766", "fair")  # percent form of the gold
    assert match_variant("Answer: 0.79 percent", "Answer: 0.8 percent", "fair")  # gold was rounded
    assert not match_variant("Answer: 0.9", "Answer: 0.93", "fair")  # answer coarser than gold
    assert not match_variant("$0", "$2.0 million", "fair")
    assert all(match_variant("Answer: 14.1%", "Answer: 0.141", v) for v in VARIANTS)  # percent form
    assert not any(match_variant("no number here", "Answer: 5", v) for v in VARIANTS)
