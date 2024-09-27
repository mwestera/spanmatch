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
    ([ABC, FG], [[ABC, FG], [KLM]], 0, (EXACT, 0, None)),
    ([FG, ABC], [[ABC, FG], [KLM]], 0, (EXACT, 0, None)),
    ([FG, ABC], [[KLM], [ABC, FG]], 0, (EXACT, 1, None)),
    ([ABC, FG], [[KLM], [BC, FG]], 0, (SUPERSPAN, None, None)),
    ([ABC, FG], [[KLM], [CDE, FG]], 0, (OVERLAP, None, None)),
    ([ABC], [[KLM], [BC]], 0, (SUPERSPAN, None, None)),
    ([BC], [[KLM], [ABC]], 0, (SUBSPAN, None, None)),
    ([ABC], [[KLM], [BC]], 2, (EXACT, 1, None)),
    ([BC], [[KLM], [ABC]], 1, (EXACT, 1, None)),
    ([ABC, FG], [[KLM], [BC, IJ]], 0, (OVERLAP, None, None)),
    ([BC, FG], [[KLM], [ABC, IJ]], 0, (OVERLAP, None, None)),
    ([ABC, FG], [[KLM], [CDE, IJ]], 0, (OVERLAP, None, None)),
    ([], [[KLM], [ABC, FG]], 0, (SUBSPAN, None, None)),
    ([], [[KLM], []], 1, (EXACT, 1, None)),
    ([ABC, FG], [[BC, FG], [KLM]], 1, (EXACT, 0, None)),
    ([ABCDE], [[KLM], [BC, IJ]], 2, (SUPERSPAN, None, None)),
    ([BC], [[KLM], [ABCDE]], 1, (SUBSPAN, None, None)),
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
    result = match_spans_categorical(span1, span2, margin=margin)
    assert result == target



span_BC = [(2, 6)]
span_CDE = [(4, 10)]
span_C = [(4, 6)]
cases_tokenwise = [
    (span_BC, span_CDE, 'a b c d e f g h i j', {'precision': 1/2, 'recall': 1/3, 'f1': 2 / 5, 'count': 10}),
    (span_BC, span_C, 'a b c d e f g h i j', {'precision': 1/2, 'recall': 1.0, 'f1': 2 / 3, 'count': 10})
]


@pytest.mark.parametrize("case", cases_tokenwise)
def test_match_spans_tokenwise(case):
    span1, span2, text, target = case
    result = match_spans_tokenwise(span1, span2, text)
    assert result == target