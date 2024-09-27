import functools
import math
from typing import List, Any
import itertools
import re

MATCH_LABELS = ['disjoint', 'overlap', 'subspan', 'superspan', 'exact']
DISJOINT, OVERLAP, SUBSPAN, SUPERSPAN, EXACT = MATCH_LABELS


Span = list[tuple[int, int]]    # span can be discontinuous, hence a list of tuples


def match_span_many_to_many(spans1: list[Span], spans2: list[Span], in_order: bool = False, margin: int = 2, text: str = None) \
        -> tuple[tuple[list[str], list[int]], tuple[list[str], list[int]], list[dict]]:
    """
    Crosses two lists of spans, matching each to its best counterpart, returning match labels (EXACT, OVERLAP, etc),
    indices of each span1s's EXACT counterpart in spans2, and, if text is provided, a list of (best) tokenwise scores for each span.
    """
    if in_order:
        match_tuples_left = [match_span_one_to_many(span1, [span2], margin=margin, text=text) for span1, span2 in zip(spans1, spans2)]
        match_tuples_right = [match_span_one_to_many(span2, [span1], margin=margin, text=text) for span2, span1 in zip(spans2, spans1)]
    else:
        match_tuples_left = [match_span_one_to_many(spans1, spans2, margin=margin, text=text) for spans1 in spans1]
        match_tuples_right = [match_span_one_to_many(span2, spans1, margin=margin, text=text) for span2 in spans2]

    match_labels_left, match_indices_left, scores_left = list(zip(*match_tuples_left))
    match_labels_right, match_indices_right, _ = list(zip(*match_tuples_right))

    return (match_labels_left, match_indices_left), (match_labels_right, match_indices_right), scores_left


def match_span_one_to_many(span1: Span, spans2: list[Span], margin=2, text=None):
    match_labels = [match_spans_categorical(span1, span2, margin=margin) for span2 in spans2]
    best_index, best_label = max(enumerate(match_labels), key=lambda t: MATCH_LABELS.index(t[1]))
    index_of_exact_match = best_index if best_label == EXACT else None

    best_scores = None

    if text is not None:
        match_scores = [match_spans_tokenwise(span1, span2, text) for span2 in spans2]
        best_index, best_scores = max(enumerate(match_scores), key=lambda t: t[1]['f1'])

    return best_label, index_of_exact_match, best_scores


def span_to_set(span: Span) -> set[int]:
    """
    >>> span_to_set([(1, 4), (7, 10)])
    {1, 2, 3, 7, 8, 9}
    >>> span_to_set([(1, 4)])
    {1, 2, 3}
    """
    return set(itertools.chain(*(range(*s) for s in span)))


def match_spans_categorical(span1: Span, span2: Span, margin=2) -> str:

    set1 = span_to_set(span1)
    set2 = span_to_set(span2)

    if set1 == set2:
        return EXACT
    if (len(set1 - set2) <= margin) and (len(set2 - set1) <= margin):
        return EXACT # NEAR_EXACT
    if len(set1 - set2) <= margin:
        return SUBSPAN
    if len(set2 - set1) <= margin:
        return SUPERSPAN
    if set1 & set2:
        return OVERLAP
    return DISJOINT


word_regex = re.compile(r'\w+')


@functools.cache
def get_token_spans(text: str) -> list[Span]:
    return [[match.span()] for match in word_regex.finditer(text)]


def match_spans_tokenwise(span1: Span, span2: Span, text: str) -> dict:

    set1 = span_to_set(span1)
    set2 = span_to_set(span2)

    predictions = []
    targets = []
    for token_span in get_token_spans(text):
        token_set = span_to_set(token_span)
        predictions.append(True if token_set <= set1 else False)
        targets.append(True if token_set <= set2 else False)

    return get_biclass_scores(predictions, targets)


def compute_categorical_scores(match_labels_left: list[str], match_labels_right: list[str]) -> dict:
    # TODO div/0
    n_exact_left = sum(l == EXACT for l in match_labels_left)
    precision = n_exact_left / len(match_labels_left)

    n_exact_right = sum(l == EXACT for l in match_labels_right)
    recall = n_exact_right / len(match_labels_right)

    f1 = harmonic_mean(precision, recall)
    count = len([d for d in match_labels_left if d is not None])
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'count': count,
    }


def get_biclass_scores(predictions: list[bool], targets: list[bool]) -> dict:
    # TODO div/0
    tp = sum(p and t for p, t in zip(predictions, targets))

    # tn = sum(not p and not t for p, t in zip(predictions, targets))
    # accuracy = (tp + tn) / len(predictions)

    precision = tp / sum(predictions)
    recall = tp / sum(targets)
    f1 = harmonic_mean(precision, recall)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'count': len(predictions)
    }


def harmonic_mean(a, b):
    if a == 0 or b == 0:
        return 0
    return 2 / (1/a + 1/b)


def nanmean(i: list):
    j = [f for f in i if not math.isnan(f)]
    if not j:
        return math.nan

    return sum(j) / len(j)
