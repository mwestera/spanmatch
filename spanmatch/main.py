import functools
import math
from typing import List, Any, Union, Iterable
import itertools
import re

MATCH_LABELS = ['disjoint', 'overlap', 'subspan', 'superspan', 'exact']
DISJOINT, OVERLAP, SUBSPAN, SUPERSPAN, EXACT = MATCH_LABELS


Span = list[tuple[int, int]]    # span can be discontinuous, hence a list of tuples


def make_alignment_mapping(spans1: list[Span], spans2: list[Span], text, by_score='f1') -> list[Union[int, None]]:

    scores = []
    for (i, span1), (j, span2) in itertools.product(enumerate(spans1), enumerate(spans2)):
        predictions, targets = match_spans_tokenwise(span1, span2, text)
        scores.append((i, j, compute_binary_classification_scores(predictions, targets)[by_score]))
    scores.sort(key=lambda t: t[-1], reverse=True)

    already_matched1 = set()
    already_matched2 = set()
    index_map = {}
    for i, j, _ in scores:
        if i in already_matched1 or j in already_matched2:
            continue
        already_matched1.add(i)
        already_matched2.add(j)
        index_map[i] = j

    # spans1 that do not have a counterpart in spans2, map to None
    for i in range(len(spans1)):
        if i not in already_matched1:
            index_map[i] = None

    return [index_map[i] for i in range(len(index_map))]


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


def match_spans_tokenwise(span1: Span, span2: Span, text: str) -> tuple[list, list]:

    set1 = span_to_set(span1)
    set2 = span_to_set(span2)

    predictions = []
    targets = []
    for token_span in get_token_spans(text):
        token_set = span_to_set(token_span)
        predictions.append(True if token_set <= set1 else False)
        targets.append(True if token_set <= set2 else False)

    return predictions, targets


def compute_categorical_scores(match_labels_left: list[str], match_labels_right: list[str]) -> dict:
    n_exact_left = sum(l == EXACT for l in match_labels_left)
    precision = n_exact_left / len(match_labels_left) if match_labels_left else float('nan')

    n_exact_right = sum(l == EXACT for l in match_labels_right)
    recall = n_exact_right / len(match_labels_right) if match_labels_right else float('nan')

    f1 = harmonic_mean(precision, recall) if match_labels_left and match_labels_right else float('nan')

    count = len([d for d in match_labels_left if d is not None])
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'count': count,
    }


def compute_binary_classification_scores(predictions: list[bool], targets: list[bool]) -> dict:
    tp = sum(p and t for p, t in zip(predictions, targets))

    # tn = sum(not p and not t for p, t in zip(predictions, targets))
    # accuracy = (tp + tn) / len(predictions)

    sumpreds = sum(predictions)
    sumtargets = sum(targets)

    precision = tp / sumpreds if sumpreds else float('nan')
    recall = tp / sumtargets if sumtargets else float('nan')
    f1 = harmonic_mean(precision, recall) if sumpreds and sumtargets else float('nan')

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'count': len(predictions)
    }


def compute_macro_scores(scores):
    scores = {
        'precision': nanmean(s['precision'] for s in scores),
        'recall': nanmean(s['recall'] for s in scores),
        'f1': nanmean(s['f1'] for s in scores),
        'count': len(scores)
    }
    return scores


def harmonic_mean(a, b):
    if a == 0 or b == 0:
        return 0
    return 2 / (1/a + 1/b)


def nanmean(i: Iterable):
    j = [f for f in i if not math.isnan(f)]
    if not j:
        return math.nan

    return sum(j) / len(j)
