import math
from typing import List, Any
import itertools


MATCH_LABELS = ['disjoint', 'overlap', 'subspan', 'superspan', 'exact']
DISJOINT, OVERLAP, SUBSPAN, SUPERSPAN, EXACT = MATCH_LABELS


def match_span_many_to_many(spans1, spans2, in_order=False, margin=2):
    if in_order:
        match_tuples_left = [match_span_one_to_many(span1, [span2], margin=margin) for span1, span2 in zip(spans1, spans2)]
        match_tuples_right = [match_span_one_to_many(span2, [span1], margin=margin) for span2, span1 in zip(spans2, spans1)]
    else:
        match_tuples_left = [match_span_one_to_many(spans1, spans2, margin=margin) for spans1 in spans1]
        match_tuples_right = [match_span_one_to_many(span2, spans1, margin=margin) for span2 in spans2]

    match_labels_left, match_indices_left = list(zip(*match_tuples_left))
    match_labels_right, match_indices_right = list(zip(*match_tuples_right))

    return (match_labels_left, match_indices_left), (match_labels_right, match_indices_right)


def match_span_one_to_many(span1, spans2, margin=2):
    """
    Compares a single spanset (= potentially discontinuous span) against multiple 'candidate spansets'.
    """
    span2_labels = [match_spans(span1, span2, margin=margin) for span2 in spans2]
    best_index, best_label = max(enumerate(span2_labels), key=lambda t: MATCH_LABELS.index(t[1]))
    index_of_exact_match = best_index if best_label == EXACT else None

    return best_label, index_of_exact_match


def span_to_set(span):
    return set(itertools.chain(*(range(*s) for s in span)))


def match_spans(span1, span2, margin=2) -> str:

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


def binarize_and_average_match_labels(match_labels, cutoff_label=EXACT):
    acceptable_labels = MATCH_LABELS[MATCH_LABELS.index(cutoff_label):]

    match_labels = list(filter(None, match_labels))
    if not match_labels:
        return 1

    surviving_labels = [m in acceptable_labels for m in match_labels]
    if not surviving_labels:
        return math.nan
    return sum(surviving_labels) / len(surviving_labels)


def harmonic_mean(a, b):
    if a == 0 or b == 0:
        return 0
    return 2 / (1/a + 1/b)


def nanmean(i):
    j = [f for f in i if not math.isnan(f)]
    if not j:
        return math.nan

    return sum(j) / len(j)


def compute_match_scores(match_labels_left, match_labels_right, cutoff_label=EXACT):
    precision = binarize_and_average_match_labels(match_labels_left, cutoff_label=cutoff_label)
    recall = binarize_and_average_match_labels(match_labels_right, cutoff_label=cutoff_label)
    f1 = harmonic_mean(precision, recall)
    count = len([d for d in match_labels_left if d is not None])
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'count': count,
    }