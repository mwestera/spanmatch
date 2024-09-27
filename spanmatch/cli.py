#!/usr/bin/python

import argparse
import sys
import json
import itertools
from .main import *

import logging

import math

OUTPUT_LABELS = 'labels'
OUTPUT_MATCHES = 'matches'
OUTPUT_SCORES_PER_LINE = 'scores'
OUTPUT_AGGREGATE_SCORES = 'score'
OUTPUT_JSONL = 'jsonl'

modes = [OUTPUT_LABELS, OUTPUT_MATCHES, OUTPUT_SCORES_PER_LINE, OUTPUT_AGGREGATE_SCORES, OUTPUT_JSONL]

# TODO: Add characterwise and tokenwise scores as well.
# TODO: Add side-by-side html view with spanviz (requires text as input, which I'll need anyway for tokenwise f1...)
# TODO: Also add sentencewise scores: if spans are expanded to the full sentence; and implement --merge

def main():

    argparser = argparse.ArgumentParser(description='Evaluating multi-disco-spans.')
    argparser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="file with text to process (default: stdin)")
    argparser.add_argument('--output', type=str, choices=modes, default=OUTPUT_LABELS, help="whether to output match labels, match orders, per-line scores, aggregate scores, or JSON structures")
    argparser.add_argument('--merge', action='store_true', help="whether to make spans continuous prior to eval")
    argparser.add_argument('--verb', action='store_true', help="verbose, set logging level to debug")
    argparser.add_argument('--ordered', action='store_true', help="if the pairs are in the same order (so don't compare all to all)")

    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verb else logging.INFO, format='')

    if args.merge:
        raise NotImplementedError('--merge not yet supported')

    pairs_to_compare = parse_lines(args.file)
    results = evaluate_all(pairs_to_compare, do_print=True, mode=args.output, ordered=args.ordered)


def evaluate_all(pairs_to_compare, do_print=False, mode=OUTPUT_AGGREGATE_SCORES, ordered=False):

    # TODO Refactor? with the help of a 'record' dict.
    # TODO: Allow html output

    print_or_log = print if do_print else logging.debug

    all_match_labels_left = []
    all_match_labels_right = []
    scores_per_doc = []

    for item in pairs_to_compare:
        (annotator1, spans1), (annotator2, spans2) = item.items()
        (match_labels_left, match_indices_left), (match_labels_right, match_indices_right), scores_tokenwise = match_span_many_to_many(spans1, spans2, in_order=ordered)

        # TODO: Do something with scores_tokenwise
        scores_categorical = compute_categorical_scores(match_labels_left, match_labels_right)

        if mode == OUTPUT_SCORES_PER_LINE:
            print_or_log('{precision:.2f},{recall:.2f},{f1:.2f}  ({count})'.format(**scores_categorical))
        elif mode == OUTPUT_LABELS:
            print_or_log(','.join(f'{l}' for l in match_labels_left) + '\t' + ','.join(f'{l}' for l in match_labels_right))
        elif mode == OUTPUT_MATCHES:
            print_or_log(','.join(f'{i}' if i is not None else '' for i in match_indices_left) + '\t' +
                         ','.join(f'{i}' if i is not None else '' for i in match_indices_right))
        elif mode == OUTPUT_JSONL:
            print_or_log(json.dumps({
                annotator1: {'labels': match_labels_left, 'indices': match_labels_left},
                annotator2: {'labels': match_labels_right, 'indices': match_indices_right},
                **scores_categorical,
            }))

        all_match_labels_left.extend(match_labels_left)
        all_match_labels_right.extend(match_labels_right)
        scores_per_doc.append(scores_categorical)

    micro_scores = compute_categorical_scores(all_match_labels_left, all_match_labels_right)

    macro_scores = {
        'precision': nanmean(s['precision'] for s in scores_per_doc),
        'recall': nanmean(s['recall'] for s in scores_per_doc),
        'f1': nanmean(s['f1'] for s in scores_per_doc),
        'count': sum(s['count'] for s in scores_per_doc),
    }
    aggregated_scores = {'micro': micro_scores, 'macro': macro_scores, }

    format_scores = '{precision:.2f}, {recall:.2f}, {f1:.2f}  ({count})'.format
    if mode == OUTPUT_AGGREGATE_SCORES:
        print(format_scores(**micro_scores))
        print(format_scores(**macro_scores))
    else:
        logging.info('Aggregated:')
        logging.info('micro: ' + format_scores(**micro_scores))
        logging.info('macro: ' + format_scores(**macro_scores))

    return aggregated_scores


def parse_lines(file):
    for line in file:
        try:
            d = json.loads(line)    # TODO: type validation (dict with two keys, values list of lists of pairs of ints
            yield d
        except json.JSONDecodeError:
            left, right = line.strip().split('\t')
            spansets1 = [[[int(i) for i in span.split('-')] for span in s.split(',')] for s in left.split(';')]
            spansets2 = [[[int(i) for i in span.split('-')] for span in s.split(',')] for s in right.split(';')]
            yield {'left': spansets1, 'right': spansets2}


if __name__ == '__main__':
    main()