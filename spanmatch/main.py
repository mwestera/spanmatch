import math

MATCH_LABELS = ['disjoint', 'overlap*', 'overlap', 'fuzzy*', 'fuzzy', 'exact*', 'exact']
DISJOINT = 0
OVERLAP = 2
FUZZY = 4
EXACT = 6



def match_span_one_to_many(span1, spans2, return_index=False):
    """
    Compares a single spanset (= potentially discontinuous span) against multiple 'candidate spansets'.
    Returns a string indicating the type of match.
    """
    span2_labels = [match_spans(span1, span2) for span2 in spans2]
    span1_label = max(span2_labels) if span2_labels else None
    index_of_exact_match = span2_labels.index(EXACT) if span1_label == EXACT else None

    span1_label_str = MATCH_LABELS[span1_label]

    if return_index:
        return span1_label_str, index_of_exact_match
    else:
        return span1_label_str


def match_span_many_to_many(spans1, spans2):
        # TODO: enable this?
        # if order_fixed:
        #     left = [match_span_multi(spanset1, [spanset2]) for spanset1, spanset2 in zip(spansets1, spansets2)]
        #     right = [match_span_multi(spanset2, [spanset1]) for spanset2, spanset1 in zip(spansets2, spansets1)]
        # else:
        match_labels_left, match_indices_left = list(zip(*[match_span_one_to_many(spans1, spans2, return_index=True) for spans1 in spans1]))
        match_labels_right, match_indices_right = list(zip(*[match_span_one_to_many(span2, spans1, return_index=True) for span2 in spans2]))
        return (match_labels_left, match_indices_left), (match_labels_right, match_indices_right)


def match_spans(span1, span2):
    # TODO: This function belongs in a span class
    subspan1_labels = []
    for subspan1 in span1:
        subspan2_labels = []
        for subspan2 in span2:
            start1, end1 = subspan1[0], subspan1[1]
            start2, end2 = subspan2[0], subspan2[1]
            if end1 < start2 or end2 < start1:
                subspan2_labels.append(DISJOINT)
            if start2 <= start1 <= end2 or start2 <= end1 <= end2:
                subspan2_labels.append(OVERLAP)
            if abs(start2 - start1) < 2 and abs(end2 - end1) < 2:
                subspan2_labels.append(FUZZY)
            if start1 == start2 and end1 == end2:
                subspan2_labels.append(EXACT)
        subspan1_labels.append(max(subspan2_labels))
    best_subspan1_label = max(subspan1_labels)
    if best_subspan1_label != min(subspan1_labels):
        best_subspan1_label -= 1  # partial
    return best_subspan1_label


def binarize_and_average_match_labels(match_labels, cutoff_label=MATCH_LABELS[EXACT]):
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


def compute_match_scores(match_labels_left, match_labels_right, cutoff_label=MATCH_LABELS[EXACT]):
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