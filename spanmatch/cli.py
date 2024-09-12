#!/usr/bin/python

import argparse
import sys
import json
import itertools
from .main import *

import logging

import math


MATCH_LABELS = ['disjoint', 'overlap*', 'overlap', 'fuzzy*', 'fuzzy', 'exact*', 'exact']
DISJOINT = 0
OVERLAP = 2
FUZZY = 4
EXACT = 6

OUTPUT_LABELS = 'labels'
OUTPUT_MATCHES = 'matches'
OUTPUT_SCORES_PER_LINE = 'scores'
OUTPUT_AGGREGATE_SCORES = 'score'
OUTPUT_JSONL = 'jsonl'

def main():

    logging.basicConfig(level=logging.INFO, format='')

    argparser = argparse.ArgumentParser(description='Evaluating multi-disco-spans.')
    argparser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="file with text to process (default: stdin)")
    argparser.add_argument('--cutoff', required=False, type=str, default=MATCH_LABELS[-1], choices=MATCH_LABELS, help="how to binarize labels by thresholding")
    argparser.add_argument('--output', type=str, choices=[OUTPUT_LABELS, OUTPUT_MATCHES, OUTPUT_SCORES_PER_LINE, OUTPUT_AGGREGATE_SCORES, OUTPUT_JSONL], default=OUTPUT_LABELS, help="whether to output match labels, match orders, per-line scores, aggregate scores, or JSON structures")
    argparser.add_argument('--merge', action='store_true', help="whether to make spans continuous prior to eval")

    args = argparser.parse_args()

    if args.merge:
        raise NotImplementedError('--merge not yet supported')
    if args.output in [OUTPUT_JSONL]:
        raise NotImplementedError(f'--output {args.output} not yet supported')

    match_labels_left_per_doc = []
    match_labels_right_per_doc = []
    precs_per_doc = []
    recs_per_doc = []
    f1s_per_doc = []

    for spans1, spans2 in read_lines(args.file, merge_multispans=args.merge):
        # TODO: Improve this
        # if order_fixed:
        #     left = [match_span_multi(spanset1, [spanset2]) for spanset1, spanset2 in zip(spansets1, spansets2)]
        #     right = [match_span_multi(spanset2, [spanset1]) for spanset2, spanset1 in zip(spansets2, spansets1)]
        # else:
        match_labels_left, match_indices_left = list(zip(*[match_span_multi(spans1, spans2, return_index=True) for spans1 in spans1]))
        match_labels_right, match_indices_right = list(zip(*[match_span_multi(span2, spans1, return_index=True) for span2 in spans2]))

        precision = binarize_and_average_match_labels(match_labels_left, cutoff_label=args.cutoff)
        recall = binarize_and_average_match_labels(match_labels_right, cutoff_label=args.cutoff)
        f1 = harmonic_mean(precision, recall)

        if args.output == OUTPUT_SCORES_PER_LINE:
            print(f'{precision:.2f},{recall:.2f},{f1:.2f}')
        elif args.output == OUTPUT_LABELS:
            print(','.join(f'{l}' for l in match_labels_left), ','.join(f'{l}' for l in match_labels_right), sep='\t')
        elif args.output == OUTPUT_MATCHES:
            print(','.join(f'{i}' if i is not None else '' for i in match_indices_left), ','.join(f'{i}' if i is not None else '' for i in match_indices_right), sep='\t')

        match_labels_left_per_doc.extend(match_labels_left)
        match_labels_right_per_doc.extend(match_labels_right)
        precs_per_doc.append(precision)
        recs_per_doc.append(recall)
        f1s_per_doc.append(f1)

    micro_prec = binarize_and_average_match_labels(match_labels_left_per_doc, cutoff_label=args.cutoff)
    micro_rec = binarize_and_average_match_labels(match_labels_right_per_doc, cutoff_label=args.cutoff)
    micro_f1 = harmonic_mean(micro_prec, micro_rec)
    count = len([d for d in itertools.chain(*match_labels_left_per_doc) if d is not None])

    macro_prec = nanmean(precs_per_doc)
    macro_rec = nanmean(recs_per_doc)
    macro_f1 = nanmean(f1s_per_doc)
    doc_count = len([d for d in f1s_per_doc if d is not None])

    agg_scores = \
        (f'{micro_prec:.2f}, {micro_rec:.2f}, {micro_f1:.2f}  ({count})\n'
         f'{macro_prec:.2f}, {macro_rec:.2f}, {macro_f1:.2f}  ({doc_count})')

    if args.output == OUTPUT_AGGREGATE_SCORES:
        print(agg_scores)
    else:
        logging.info(agg_scores)


def read_lines(file, merge_multispans=False):
    # TODO handle merge_multispans
    for line in file:
        left, right = line.strip().split('\t')
        spansets1 = [[[int(i) for i in span.split('-')] for span in s.split(',')] for s in left.split(';')]
        spansets2 = [[[int(i) for i in span.split('-')] for span in s.split(',')] for s in right.split(';')]

        # TODO: alternative: json, assuming pairs OR start/end keys.

        yield spansets1, spansets2


def match_span_multi(span1, spans2, return_index=False):
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


def binarize_and_average_match_labels(match_labels, cutoff_label='exact', partial=False):
    if partial:
        cutoff_label += '*'
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


if __name__ == '__main__':
    main()