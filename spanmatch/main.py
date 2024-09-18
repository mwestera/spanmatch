import math

MATCH_LABELS = ['disjoint', 'overlap*', 'overlap', 'fuzzy*', 'fuzzy', 'exact*', 'exact']
DISJOINT = 0
OVERLAP = 2
FUZZY = 4
EXACT = 6



def match_span_one_to_many(span1, spans2):
    """
    Compares a single spanset (= potentially discontinuous span) against multiple 'candidate spansets'.
    """
    span2_labels = [match_spans(span1, span2) for span2 in spans2]
    span1_label = max(span2_labels)
    index_of_exact_match = span2_labels.index(EXACT) if span1_label == EXACT else None

    span1_label_str = MATCH_LABELS[span1_label]

    return span1_label_str, index_of_exact_match


def match_span_many_to_many(spans1, spans2, in_order=False):
        if in_order:
            match_labels_left, match_indices_left = list(zip(*[match_span_one_to_many(span1, [span2]) for span1, span2 in zip(spans1, spans2)]))
            match_labels_right, match_indices_right = list(zip(*[match_span_one_to_many(span2, [span1]) for span2, span1 in zip(spans2, spans1)]))
        else:
            match_labels_left, match_indices_left = list(zip(*[match_span_one_to_many(spans1, spans2) for spans1 in spans1]))
            match_labels_right, match_indices_right = list(zip(*[match_span_one_to_many(span2, spans1) for span2 in spans2]))
        return (match_labels_left, match_indices_left), (match_labels_right, match_indices_right)


def match_spans(span1, span2):
    # TODO: This function belongs in a span class
    subspan1_labels = []
    if not span1:
        return DISJOINT if span2 else EXACT

    for subspan1 in span1:
        subspan2_labels = [DISJOINT]
        for subspan2 in span2:
            start1, end1 = subspan1[0], subspan1[1]
            start2, end2 = subspan2[0], subspan2[1]
            if start1 == start2 and end1 == end2:
                subspan2_labels.append(EXACT)
            elif abs(start2 - start1) < 2 and abs(end2 - end1) < 2:
                subspan2_labels.append(FUZZY)
            elif end1 < start2 or end2 < start1:
                subspan2_labels.append(DISJOINT)
            elif start2 <= start1 <= end2 or start2 <= end1 <= end2 or start1 <= start2 <= end1 or start1 <= end2 <= end1:
                subspan2_labels.append(OVERLAP)
            else:
                raise NotImplementedError('Hmmm I thought I covered all cases...')
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