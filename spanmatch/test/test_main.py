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
def test_match_spans_categorical(case):

    span1, span2, margin, target = case
    result = match_spans_categorical(span1, span2, margin=margin)
    assert result == target


span_BC = [(2, 6)]
span_CDE = [(4, 10)]
span_C = [(4, 6)]
cases_tokenwise = [
    (span_BC, span_CDE, 'a b c d e f g h i j', (
        [False, True, True, False, False, False, False, False, False, False],
        [False, False, True, True, True, False, False, False, False, False])),
    (span_BC, span_C, 'a b c d e f g h i j', (
        [False, True, True, False, False, False, False, False, False, False],
        [False, False, True, False, False, False, False, False, False, False]))
]


@pytest.mark.parametrize("case", cases_tokenwise)
def test_match_spans_tokenwise(case):
    span1, span2, text, target = case
    result = match_spans_tokenwise(span1, span2, text)
    assert result == target

cases_scores = [
    ([True, True, False], [True, True, True], {'precision': 1.0, 'recall': 2/3, 'f1': 0.8, 'count': 3,}),
    ([False, True, True, False, False, False, False, False, False, False], [False, False, True, False, False, False, False, False, False, False], {'precision': 1/2, 'recall': 1.0, 'f1': 2 / 3, 'count': 10}),
    ([False, True, True, False, False, False, False, False, False, False], [False, False, True, True, True, False, False, False, False, False], {'precision': 1/2, 'recall': 1/3, 'f1': 2 / 5, 'count': 10}),
]

@pytest.mark.parametrize("case", cases_scores)
def test_compute_binary_classification_scores(case):
    preds, targs, goal = case
    scores = compute_binary_classification_scores(preds, targs)
    assert scores == goal