from spanmatch.main import *
import pytest

# ABCDEFGHIJ K  L  M  N
# 0123456789 10 11 12 13

ABC = (0, 3)
ABCDE = (0, 5)
BC = (1, 3)
FG = (5, 7)
KLM = (10, 13)
CDE = (2, 5)
IJ = (8, 10)

cases_multi = [
    ([ABC, FG], [[ABC, FG], [KLM]], 0, (EXACT, 0)),
    ([FG, ABC], [[ABC, FG], [KLM]], 0, (EXACT, 0)),
    ([FG, ABC], [[KLM], [ABC, FG]], 0, (EXACT, 1)),
    ([ABC, FG], [[KLM], [BC, FG]], 0, (SUPERSPAN, None)),
    ([ABC, FG], [[KLM], [CDE, FG]], 0, (OVERLAP, None)),
    ([ABC], [[KLM], [BC]], 0, (SUPERSPAN, None)),
    ([BC], [[KLM], [ABC]], 0, (SUBSPAN, None)),
    ([ABC], [[KLM], [BC]], 2, (EXACT, 1)),
    ([BC], [[KLM], [ABC]], 1, (EXACT, 1)),
    ([ABC, FG], [[KLM], [BC, IJ]], 0, (OVERLAP, None)),
    ([BC, FG], [[KLM], [ABC, IJ]], 0, (OVERLAP, None)),
    ([ABC, FG], [[KLM], [CDE, IJ]], 0, (OVERLAP, None)),
    ([], [[KLM], [ABC, FG]], 0, (SUBSPAN, None)),
    ([], [[KLM], []], 1, (EXACT, 1)),
    ([ABC, FG], [[BC, FG], [KLM]], 1, (EXACT, 0)),
    ([ABCDE], [[KLM], [BC, IJ]], 2, (SUPERSPAN, None)),
    ([BC], [[KLM], [ABCDE]], 1, (SUBSPAN, None)),
]


@pytest.mark.parametrize("case", cases_multi)
def test_match_span_one_to_many(case):

    """
    if one is exact, then it goes to the minimum, but not below overlap.
    """
    one_span, many_spans, margin, target = case
    result = match_span_one_to_many(one_span, many_spans, margin=margin)
    assert result == target


cases_single = [
    ([ABC, FG], [ABC, FG], 0, EXACT),
    ([FG, ABC], [ABC, FG], 0, EXACT),
    ([FG, ABC], [ABC, FG], 0, EXACT),
    ([ABC, FG], [BC, FG], 0, SUPERSPAN),
    ([ABC, FG], [CDE, FG], 0, OVERLAP),
    ([ABC], [BC], 0, SUPERSPAN),
    ([BC], [ABC], 0, SUBSPAN),
    ([ABC], [BC], 2, EXACT),
    ([BC], [ABC], 1, EXACT),
    ([ABC, FG], [BC, IJ], 0, OVERLAP),
    ([BC, FG], [ABC, IJ], 0, OVERLAP),
    ([ABC, FG], [CDE, IJ], 0, OVERLAP),
    ([], [ABC, FG], 0, SUBSPAN),
    ([], [], 0, EXACT),
    ([ABC, FG], [BC, FG], 1, EXACT),
    ([ABCDE], [BC, IJ], 2, SUPERSPAN),
    ([BC], [ABCDE], 1, SUBSPAN),
]

@pytest.mark.parametrize("case", cases_single)
def test_match_spans(case):

    span1, span2, margin, target = case
    result = match_spans(span1, span2, margin=margin)
    assert result == target
